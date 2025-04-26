from flask import Flask, Response, request, jsonify
import sys
import os
import logging
from embedding_oper import load_vector_store, query_vector_store
from flask_cors import CORS
import markdown
import html2text
import json
from collections import deque
import time

app = Flask(__name__)
CORS(app) 

from openai import OpenAI
client = OpenAI(api_key="sk-a1e114714f2b4580a00cdf9fcf981743", base_url="https://api.deepseek.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 存储对话历史，使用字典存储每个会话的历史记录
conversation_history = {}
# 存储会话最后活动时间
session_last_active = {}

# 会话过期时间（秒）
SESSION_EXPIRY = 3600  # 1小时

# 清理过期会话
def cleanup_expired_sessions():
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, last_active in session_last_active.items()
        if current_time - last_active > SESSION_EXPIRY
    ]
    
    for session_id in expired_sessions:
        del conversation_history[session_id]
        del session_last_active[session_id]
        logger.info(f"会话 {session_id} 已过期并清理")

# 构建RAG提示词
def build_rag_prompt(query1, context_docs, history=None):
    try:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = '你是一个无比专业的重庆市源著天街商场的客服，客人询问的问题，你在上下文中的Q中进行定位，并直接返回A中的信息，不允许随意编造。'
        prompt += f"上下文信息：\n{context}\n\n"
        
        # 添加历史对话
        if history:
            prompt += "历史对话：\n"
            for msg in history:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "\n"
            
        prompt += f"问题：{query1}\n\n"
        prompt += '请基于上述上下文信息, 返回准确无误的回答，不准包含"A："。如果无法得到答案，则返回"不好意思，我不太清楚您的问题"，不允许在答案中添加编造成分。'
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
@app.route('/api/workstation/agent', methods=['POST','GET'])
def echo():
    k = 5
    store_name = "extra"
    
    try:
        # 获取POST请求的JSON数据
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400

        # 从请求体中获取查询、会话ID和历史记录
        query = data.get('query')
        session_id = data.get('session_id', 'default')
        history = data.get('history', [])

        if not query:
            return jsonify({"error": "缺少query参数"}), 400

        # 清理过期会话
        cleanup_expired_sessions()

        # 更新会话最后活动时间
        session_last_active[session_id] = time.time()

        # 检索相关文档
        retriever = global_retriever if global_retriever else load_vector_store(store_name, k)
        context_docs = query_vector_store(retriever, query)
        
        # 构建提示词，包含历史对话
        prompt = build_rag_prompt(query, context_docs, history)
        
        # 调用模型 - 非流式处理
        try:
            response = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{"role": "user", "content": prompt}],
                temperature=1.3,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content

            # 更新对话历史
            if session_id not in conversation_history:
                conversation_history[session_id] = deque(maxlen=10)  # 限制历史记录长度
            
            conversation_history[session_id].append({"role": "user", "content": query})
            conversation_history[session_id].append({"role": "assistant", "content": content})

            # 返回完整响应
            return jsonify({
                "content": content,
                "status": "completed",
                "history": list(conversation_history[session_id])
            })
                
        except Exception as e:
            logger.error(f"调用模型时发生错误: {str(e)}")
            return jsonify({"error": str(e)}), 500
                    
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 本地快捷测试入口
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
