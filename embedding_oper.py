import numpy as np
import pickle
import os
import logging
from langchain_core.documents import Document
import time
from rank_bm25 import BM25Okapi
import jieba
from typing import List

# 设置环境变量以解决tokenizers并行处理问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BM25Store:
    def __init__(self, meta_path="extra/metadata.pkl"):
        self.meta_path = meta_path
        self.metadata = []
        self.bm25 = None
        self.tokenized_corpus = None
        
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        
        self._load_or_create_store()
        
    def _load_or_create_store(self):
        """加载或创建存储"""
        if os.path.exists(self.meta_path):
            logger.info(f"加载已有元数据 {self.meta_path}")
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            self._init_bm25()
        else:
            logger.info("创建新的BM25存储")
            
    def _init_bm25(self):
        """初始化BM25索引"""
        if not self.metadata:
            raise ValueError("Cannot initialize BM25 with empty metadata")
            
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.metadata]
        if not self.tokenized_corpus:
            raise ValueError("Tokenization resulted in empty corpus")
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def add_texts(self, texts):
        """批量添加文本"""
        start_time = time.time()
        logger.info(f"添加 {len(texts)} 个文本到BM25存储")
        
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
        
        # 保存元数据
        self.metadata.extend(processed_texts)
        
        # 更新BM25索引
        self._init_bm25()
        
        # 持久化存储
        self._save()
        logger.info(f"成功添加 {len(processed_texts)} 个文本到BM25存储，耗时: {time.time() - start_time:.2f}秒")
            
    def _get_context(self, text: str, window_size: int = 2) -> str:
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
            
    def bm25_search(self, query: str, k: int = 10, window_size: int = 2) -> List[Document]:
        """使用BM25进行精确检索"""
        try:
            # BM25检索
            tokenized_query = list(jieba.cut(query))
            if not self.bm25:
                logger.error("BM25索引未初始化")
                return []
                
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_k = np.argsort(bm25_scores)[-k:][::-1]  # 获取前k个结果
            
            # 创建Document对象列表
            documents = []
            for idx in bm25_top_k:
                if idx >= len(self.metadata):
                    continue
                    
                text = self.metadata[idx]
                
                # 获取上下文
                context = self._get_context(text, window_size)
                
                doc = Document(
                    page_content=context,
                    metadata={
                        'score': float(bm25_scores[idx]),
                        'original_text': text
                    }
                )
                documents.append(doc)
        
            return documents
        except Exception as e:
            logger.error(f"BM25搜索失败: {str(e)}")
            return []
            
    def _save(self):
        """保存元数据"""
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

def load_file(file):
    """加载文件"""
    from prepare_data import load_file as load_file_fn
    return load_file_fn(file)
    
def chunk_data(data):
    """对文档进行分块"""
    from prepare_data import chunk_data as chunk_data_fn
    return chunk_data_fn(data)


class BM25Retriever:
    def __init__(self, bm25store, search_kwargs):
        self.bm25store = bm25store
        self.search_kwargs = search_kwargs
    
    def invoke(self, query, window_size=2):
        """BM25检索接口"""
        k = self.search_kwargs.get("k", 10)
        return self.bm25store.bm25_search(query, k=k, window_size=window_size)

def load_vector_store(store_name, k, docs=None):
    """
    加载BM25存储
    :param store_name: 存储目录名
    :param k: 返回的相似文档数量
    :param alpha: 不再使用，保留参数以兼容接口
    :param docs: 如果存储不存在，则使用这些文档创建
    :return: 检索器
    """
    try:
        # 确保目录存在
        os.makedirs(store_name, exist_ok=True)
        
        meta_path = f"{store_name}/metadata.pkl"
        
        # 检查元数据文件是否存在
        if not os.path.exists(meta_path):
            if docs is None:
                raise ValueError("BM25存储不存在且未提供文档，无法创建新的存储")
            logger.info("BM25存储不存在，创建新的存储...")
            store = BM25Store(meta_path=meta_path)
            store.add_texts(docs)
        else:
            store = BM25Store(meta_path=meta_path)
        
        # 返回封装好的检索器
        return BM25Retriever(store, search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"加载BM25存储失败: {str(e)}")
        raise

def query_vector_store(retriever, query):
    """
    使用BM25进行查询
    :param retriever: 检索器
    :param query: 查询文本
    :return: 检索结果
    """
    try:
        start_time = time.time()  # 添加开始时间
        logger.info(f"执行查询: {query}")
        retrieved_docs = retriever.invoke(query)
        
        elapsed_time = time.time() - start_time  # 计算耗时
        logger.info(f"检索到 {len(retrieved_docs)} 个相关文档，耗时: {elapsed_time:.2f}秒")
        
        return retrieved_docs
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise

if __name__ == "__main__":
    from prepare_data import load_file, chunk_data
    
    logger.info("加载文档...")
    data = load_file('extraKnowledge.txt')
    logger.info(f"加载到 {len(data)} 个文档")
    
    chunks = chunk_data(data,300,50)
    logger.info(f"将文档分割成 {len(chunks)} 个块")
    
    if not chunks:
        logger.error("没有生成任何文本块，请检查输入文件是否正确")
        exit(1)
        
    store_name = "extra"
    retriever = load_vector_store(store_name, 3, chunks)
    
    query = "深圳南山的数据"
    results = query_vector_store(retriever, query)
    
    print(f"\n查询: {query}")
    print(f"\n检索到 {len(results)} 个结果:")
    for i, doc in enumerate(results):
        print(f"\n文档 {i + 1}:")
        print(f"相似度: {doc.metadata.get('score', 'N/A')}")
        print(f"内容: {doc.page_content}")