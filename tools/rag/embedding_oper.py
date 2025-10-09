import numpy as np
import faiss
import pickle
import os
import logging
from langchain_core.documents import Document
import time
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Tuple

# 设置环境变量以解决tokenizers并行处理问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI
client = OpenAI(api_key="1924642822859407379", base_url="https://aigc.sankuai.com/v1/openai/native")

class OptimizedVectorStore:
    def __init__(self, index_path="extra/faiss_index", meta_path="extra/metadata.pkl", use_gpu=False):
        self.index_path = index_path
        self.meta_path = meta_path
        self.metadata = []
        self.index = None
        self.dimension = 3072  # text-embedding-3-large的维度
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.embedding_cache = {}  # 缓存嵌入向量
        self.executor = ThreadPoolExecutor(max_workers=4)  # 用于并行处理
        self.bm25 = None
        self.tokenized_corpus = None
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        self._load_or_create_index()
        
    def _load_or_create_index(self):
        """加载或创建索引"""
        if os.path.exists(self.index_path):
            logger.info(f"加载已有索引 {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            logger.info("创建新索引")
            # 使用IVF索引来加速查询 (需要先训练)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            # 标记为需要训练
            self.needs_training = True
            
        # 如果使用GPU，将索引移至GPU
        if self.use_gpu:
            logger.info("将索引移至GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        # 设置搜索范围以提高速度/精度平衡
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 10
            
        # 初始化BM25
        if self.metadata:
            self._init_bm25()
            
    def _init_bm25(self):
        """初始化BM25索引"""
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.metadata]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def _get_embedding(self, text):
        """获取单个文本的嵌入向量，带缓存"""
        # 检查缓存
        if text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # 确保text是字符串
        if hasattr(text, 'page_content'):
            text = text.page_content
        elif not isinstance(text, str):
            text = str(text)
            
        # 调用API获取嵌入向量
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # 缓存结果 (只缓存较短的文本，避免内存占用过大)
            if len(text) < 1000:
                self.embedding_cache[text] = embedding
                
            return embedding
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {str(e)}")
            # 返回零向量作为备选
            return np.zeros(self.dimension, dtype=np.float32)
            
    def _get_embeddings_batch(self, texts, batch_size=32):
        """批量获取嵌入向量，提高API效率"""
        all_embeddings = []
        
        # 批量处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # 过滤掉已缓存的，只请求未缓存的
            uncached = [t for t in batch if t not in self.embedding_cache]
            
            if uncached:
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=uncached
                    )
                    # 更新缓存
                    for j, emb_data in enumerate(response.data):
                        text = uncached[j]
                        if len(text) < 1000:  # 只缓存较短文本
                            self.embedding_cache[text] = np.array(emb_data.embedding, dtype=np.float32)
                except Exception as e:
                    logger.error(f"批量获取嵌入向量失败: {str(e)}")
            
            # 合并当前批次的嵌入向量（包括缓存的和新请求的）
            batch_embeddings = [self.embedding_cache.get(t, np.zeros(self.dimension, dtype=np.float32)) 
                               for t in batch]
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
    
    def add_texts(self, texts):
        """批量添加文本"""
        start_time = time.time()
        logger.info(f"添加 {len(texts)} 个文本到向量库")
        
        # 预处理文本
        processed_texts = []
        for text in texts:
            if hasattr(text, 'page_content'):
                content = text.page_content
            elif isinstance(text, str):
                content = text
            else:
                content = str(text)
            processed_texts.append(content)
        
        # 批量获取嵌入向量
        embeddings = self._get_embeddings_batch(processed_texts)
        
        if not embeddings:
            logger.warning("没有成功处理任何文本")
            return
            
        # 转换为numpy数组
        vectors = np.vstack(embeddings).astype(np.float32)
        
        # 检查是否需要训练索引
        if hasattr(self, 'needs_training') and self.needs_training and hasattr(self.index, 'train'):
            logger.info("训练索引...")
            self.index.train(vectors)
            self.needs_training = False
        
        # 添加到索引
        self.index.add(vectors)
        
        # 保存元数据
        self.metadata.extend(processed_texts)
        
        # 更新BM25索引
        self._init_bm25()
        
        # 持久化存储
        self._save()
        logger.info(f"成功添加 {len(processed_texts)} 个文本到向量库，耗时: {time.time() - start_time:.2f}秒")
    
    def search(self, query, k):
        """
        相似性搜索 - 优化版
        返回Document对象列表
        """
        start_time = time.time()
        try:
            # 获取查询向量
            query_vec = self._get_embedding(query).reshape(1, -1)
            
            # 执行搜索
            distances, indices = self.index.search(query_vec, k)
            
            # 创建Document对象
            documents = []
            for i, d in zip(indices[0], distances[0]):
                if i < len(self.metadata) and i >= 0:
                    text = self.metadata[i]
                    doc = Document(
                        page_content=text,
                        metadata={"score": float(d)}
                    )
                    documents.append(doc)
            
            logger.info(f"找到 {len(documents)} 个相关文档，耗时: {time.time() - start_time:.2f}秒")
            return documents
        except Exception as e:
            logger.error(f"搜索时出错: {str(e)}")
            return []
            
    def _get_context(self, text: str, window_size: int) -> str:
        """获取上下文"""
        try:
            # 在所有文档中查找当前文本
            idx = self.metadata.index(text)
            start = max(0, idx - window_size)
            end = min(len(self.metadata), idx + window_size + 1)
            
            context = "\n".join(self.metadata[start:end])
            return context
        except ValueError:
            return text
            
    def hybrid_search(self, query: str, k: int, alpha: float, 
                     window_size: int) -> List[Document]:
        try:
            # 1. BM25检索
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_k = np.argsort(bm25_scores)[-k*2:][::-1]  # 获取更多候选
            
            # 2. 向量相似度检索
            query_vec = self._get_embedding(query).reshape(1, -1)
            distances, indices = self.index.search(query_vec, k*2)  # 获取更多候选
            
            # 3. 合并结果并重排序
            candidates = {}
            
            # 添加BM25结果
            for idx in bm25_top_k:
                if idx >= len(self.metadata):
                    continue
                score = bm25_scores[idx]
                candidates[idx] = {
                    'text': self.metadata[idx],
                    'bm25_score': score,
                    'vector_score': None
                }
                
            # 添加向量检索结果
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= len(self.metadata):
                    continue
                vector_score = 1 / (1 + distance)  # 转换距离为相似度
                if idx in candidates:
                    candidates[idx]['vector_score'] = vector_score
                else:
                    candidates[idx] = {
                        'text': self.metadata[idx],
                        'bm25_score': bm25_scores[idx],
                        'vector_score': vector_score
                    }
                    
            # 归一化分数
            bm25_scores_list = [c['bm25_score'] for c in candidates.values() if c['bm25_score'] is not None]
            vector_scores_list = [c['vector_score'] for c in candidates.values() if c['vector_score'] is not None]
            
            if bm25_scores_list:
                bm25_min, bm25_max = min(bm25_scores_list), max(bm25_scores_list)
                bm25_range = bm25_max - bm25_min + 1e-6
            if vector_scores_list:
                vec_min, vec_max = min(vector_scores_list), max(vector_scores_list)
                vec_range = vec_max - vec_min + 1e-6
                
            # 计算综合得分并排序
            results = []
            for idx, info in candidates.items():
                bm25_norm = ((info['bm25_score'] - bm25_min) / bm25_range) if info['bm25_score'] is not None else 0
                vector_norm = ((info['vector_score'] - vec_min) / vec_range) if info['vector_score'] is not None else 0
                
                combined_score = alpha * vector_norm + (1 - alpha) * bm25_norm
                
                # 获取上下文
                context = self._get_context(info['text'], window_size)
                
                results.append({
                    'idx': idx,
                    'text': info['text'],
                    'context': context,
                    'score': combined_score,
                    'bm25_score': bm25_norm,
                    'vector_score': vector_norm
                })
                
            # 排序并返回前k个结果
            results.sort(key=lambda x: x['score'], reverse=True)
            
            documents = []
            for r in results[:k]:
                doc = Document(
                    page_content=r['context'],  # 使用带上下文的文本
                    metadata={
                        'score': float(r['score']),
                        'bm25_score': float(r['bm25_score']),
                        'vector_score': float(r['vector_score']),
                        'original_text': r['text']  # 保存原始文本
                    }
                    )
                documents.append(doc)
        
            return documents
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            return []
            
    def _save(self):
        """保存索引和元数据"""
        # 如果在GPU上，先移回CPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, self.index_path)
        else:
            faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

def load_file(file):
    """加载文件"""
    from tools.rag.prepare_data import load_file as load_file_fn
    return load_file_fn(file)
    
def chunk_data(data):
    """对文档进行分块"""
    from tools.rag.prepare_data import chunk_data as chunk_data_fn
    return chunk_data_fn(data)


class OptimizedVectorStoreRetriever:
    def __init__(self, vectorstore, search_kwargs, alpha):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
        self.alpha = alpha
    
    def invoke(self, query, window_size=2):
        """支持混合搜索的检索接口"""
        k = self.search_kwargs.get("k", 10)
        return self.vectorstore.hybrid_search(query, k=k, alpha=self.alpha, 
                                           window_size=window_size)

def load_vector_store(k, alpha, docs=None, store_name="extra"):
    """
    加载向量库
    :param store_name: 向量库目录名
    :param k: 返回的相似文档数量
    :param alpha: 混合搜索的alpha值
    :param docs: 如果向量库不存在，则使用这些文档创建
    :return: 检索器
    """
    try:
        # 确保目录存在
        os.makedirs(store_name, exist_ok=True)
        
        index_path = f"{store_name}/faiss_index"
        meta_path = f"{store_name}/metadata.pkl"
        
        # 检测是否有GPU支持
        use_gpu = faiss.get_num_gpus() > 0
        logger.info(f"FAISS GPU支持: {'可用' if use_gpu else '不可用'}")
        
        # 检查索引文件是否存在
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            if docs is None:
                raise ValueError("向量库不存在且未提供文档，无法创建新的向量库")
            logger.info("向量库不存在，创建新的向量库...")
            vs = OptimizedVectorStore(
                index_path=index_path,
                meta_path=meta_path,
                use_gpu=use_gpu
            )
            vs.add_texts(docs)
        else:
            vs = OptimizedVectorStore(
                index_path=index_path,
                meta_path=meta_path,
                use_gpu=use_gpu
            )
        
        # 返回封装好的检索器
        return OptimizedVectorStoreRetriever(vs, search_kwargs={"k": k}, alpha=alpha)
    except Exception as e:
        logger.error(f"加载向量库失败: {str(e)}")
        raise

def query_vector_store(retriever, query):
    """
    使用向量库进行查询
    :param retriever: 检索器
    :param query: 查询文本
    :return: 检索结果
    """
    try:
        logger.info(f"执行查询: {query}")
        retrieved_docs = retriever.invoke(query)
        logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")
        
        filtered_docs = []
        for i, doc in enumerate(retrieved_docs):
            if not hasattr(doc, 'page_content'):
                if isinstance(doc, tuple) and len(doc) >= 1:
                    text, score = doc if len(doc) >= 2 else (doc[0], 0.0)
                    if score >= 0.4: 
                        filtered_docs.append(Document(
                            page_content=text,
                            metadata={"score": score}
                        ))
            else:
                score = doc.metadata.get('score', 0.0)
                if score >= 0.4:  
                    filtered_docs.append(doc)
        
        logger.info(f"过滤后保留 {len(filtered_docs)} 个相关文档")
        return filtered_docs
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise

if __name__ == "__main__":
    from tools.rag.prepare_data import load_file, chunk_data
    
    logger.info("加载文档...")
    data = load_file('data.txt')
    chunks = chunk_data(data,300,50)
    
    retriever = load_vector_store(5, 0.2, chunks)
    
    query = "深圳南山的数据"
    results = query_vector_store(retriever, query)
    
    print(f"\n查询: {query}")
    print(f"\n检索到 {len(results)} 个结果:")
    for i, doc in enumerate(results):
        print(f"\n文档 {i + 1}:")
        print(f"相似度: {doc.metadata.get('score', 'N/A')}")
        print(f"内容: {doc.page_content}")