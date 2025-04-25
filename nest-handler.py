from flask import Flask, Response, stream_with_context, request, jsonify
import sys
import os
import logging
from embedding_oper import load_vector_store, query_vector_store
from flask_cors import CORS
import markdown
import html2text
import json

app = Flask(__name__)
CORS(app) 

from openai import OpenAI
client = OpenAI(api_key="sk-a1e114714f2b4580a00cdf9fcf981743", base_url="https://api.deepseek.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 构建RAG提示词
def build_rag_prompt(query1, context_docs):
    try:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = '你是一个无比专业的重庆市源著天街商场的客服，客人询问的问题，你在上下文中的Q中进行定位，并直接返回A中的信息，不允许随意编造。'
        prompt += f"上下文信息：\n{context}\n\n"
        prompt += f"问题：{query1}\n\n"
        prompt += '请基于上述上下文信息, 返回准确无误的回答，不准包含“A：”。如果无法得到答案，则返回"不好意思，我不太清楚您的问题"，不允许在答案中添加编造成分。'
        return prompt
    except Exception as e:
        logger.error(f"构建提示词失败: {str(e)}")
        raise

# 将markdown模式转换为纯文本
def markdown_to_text(markdown_text):
    html = markdown.markdown(markdown_text)
    
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True  
    text_maker.ignore_images = True  
    text_maker.ignore_emphasis = True  
    
    plain_text = text_maker.handle(html)
    return plain_text

# 格式化SSE消息
def format_sse(data, event=None, id=None, retry=None):
    message = ""
    if id is not None:
        message += f"id: {id}\n"
    if event is not None:
        message += f"event: {event}\n"
    if retry is not None:
        message += f"retry: {retry}\n"
    
    if isinstance(data, dict):
        data = json.dumps(data)
    
    # 确保数据中的换行符被正确处理
    for line in data.split('\n'):
        message += f"data: {line}\n"
    
    return message + "\n"


def warmup_retriever():
    try:
        logger.info("预热检索器...")
        store_name = "extra"
        k = 5

        # 提前加载检索器
        retriever = load_vector_store(store_name, k)
        return retriever
    except Exception as e:
        logger.error(f"检索器预热失败: {str(e)}")
        return None

# 应用启动时预热检索器
global_retriever = warmup_retriever()

# '/api/python/demo/' 需要和Nest HTTP触发器对应
@app.route('/api/workstation/agent', methods=['POST'])
def echo():
    k = 5
    store_name = "extra"
    
    def generate():
        try:
            message_id = 1

            # 获取POST请求的JSON数据
            data = request.get_json(silent=True)
            if not data:
                error_msg = {"error": "请求体不能为空"}
                yield format_sse(error_msg, event="error", id=message_id)
                return

            # 从请求体中获取查询和模型
            query = data.get('query')

            if not query:
                error_msg = {"error": "缺少query参数"}
                yield format_sse(error_msg, event="error", id=message_id)
                return

            # 发送处理开始事件
            yield format_sse({"status": "开始处理请求"}, event="start", id=message_id)
            message_id += 1

            # 检索相关文档
            retriever = global_retriever if global_retriever else load_vector_store(store_name, k)
            context_docs = query_vector_store(retriever, query)
            
            # 发送生成开始事件
            yield format_sse({"status": "开始生成回答..."}, event="generation", id=message_id)
            message_id += 1
            
            # 构建提示词
            prompt = build_rag_prompt(query, context_docs)
            
            # 调用模型 - 使用健壮的流式处理
            accumulated_text = ""
            try:
                response = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.3,
                    max_tokens=2000,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        accumulated_text += chunk.choices[0].delta.content
                        yield format_sse({"content": chunk.choices[0].delta.content}, event="chunk", id=message_id)
                        message_id += 1

                # 发送完成事件
                yield format_sse({"content": accumulated_text, "status": "completed"}, event="done", id=message_id)
                
            except Exception as e:
                logger.error(f"调用模型时发生错误: {str(e)}")
                yield format_sse({"error": str(e)}, event="error", id=message_id)
                    
        except Exception as e:
            logger.error(f"处理请求时发生错误: {str(e)}")
            yield format_sse({"error": str(e)}, event="error", id=message_id)
            
    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
                   })

# wsgi标准入口，文件nest-handler.py 必须提供一个形如如下格式的handler。
def handler(environ, start_response):
    return app(environ, start_response)


# 本地快捷测试入口
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
