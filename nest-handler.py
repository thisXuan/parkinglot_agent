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
import requests

app = Flask(__name__)
CORS(app) 

from openai import OpenAI
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

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

# 修改查询商铺位置的函数
def find_store_location(store_name):
    """通过API查询商铺在商场的哪一层"""
    try:
        # 发起GET请求获取商铺信息
        url = f"http://localhost:8081/store/queryStoreInfo?query={store_name}"
        response = requests.get(url)
        
        # 解析响应
        result = response.json()
        
        if result.get("code") == 200 and result.get("data"):
            # 获取第一个匹配的店铺信息
            store = result["data"][0]
            floor = store.get("floorNumber")
            store_name = store.get("storeName")
            
            if floor is not None:
                floor_text = f"{floor}楼"
                return {"name": store_name, "floor": floor_text}
        
        # 如果请求失败或没有数据，返回None
        return {"name": store_name, "floor": None}
    except Exception as e:
        logger.error(f"查询商铺位置失败: {str(e)}")
        return {"name": store_name, "floor": None}

# 构建RAG提示词
def build_rag_prompt(query1, context_docs, history=None):
    try:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        prompt = '你是一个无比专业、很有礼貌的重庆市源著天街商场的客服，客人询问的问题，你在上下文信息中进行检索，不允许随意编造。'
        prompt += f"上下文信息：\n{context}\n\n"
        
        # 添加历史对话
        if history:
            prompt += "历史对话：\n"
            for msg in history:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "\n"
            
        prompt += f"问题：{query1}\n\n"
        prompt += '''请基于上述上下文信息, 返回准确的回答。如果无法得到答案，则返回"不好意思，我不太清楚您的问题"，不允许在答案中添加编造成分。
        
如果用户询问某个商铺在几楼，你可以调用函数查询。请注意分析问题，如果是问商铺位置，一定要调用函数而不是自己回答。'''
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

# 定义function calling的函数工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "find_store_location",
            "description": "查询商铺在商场的哪一层",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_name": {
                        "type": "string",
                        "description": "商铺名称"
                    }
                },
                "required": ["store_name"]
            }
        }
    }
]

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
        
        # 调用模型 - 使用function calling
        try:
            response = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{"role": "user", "content": prompt}],
                temperature=1.3,
                max_tokens=2000,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            content = ""
            
            # 检查是否有工具调用
            if response_message.tool_calls:
                # 处理工具调用
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if function_name == "find_store_location":
                        try:
                            store_result = find_store_location(function_args.get("store_name"))
                            
                            # 准备函数调用结果
                            function_response = "没有找到该商铺信息。"
                            if store_result["floor"]:
                                store_name = store_result["name"]
                                floor = store_result["floor"]
                                function_response = f"{store_name}在{floor}。"
                                
                            logger.info(f"商铺查询结果: {store_result}")
                        except Exception as e:
                            logger.error(f"处理商铺位置查询时发生错误: {str(e)}")
                            function_response = "查询商铺位置时发生错误，请稍后重试。"
                        
                        # 创建包含函数调用结果的新消息
                        messages = [
                            {"role": "user", "content": prompt},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": tool_call.function.arguments
                                        }
                                    }
                                ]
                            },
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": function_response
                            }
                        ]
                        
                        # 调用模型生成最终回复
                        second_response = client.chat.completions.create(
                            model='deepseek-chat',
                            messages=messages,
                            temperature=1.0,
                            max_tokens=2000
                        )
                        
                        content = second_response.choices[0].message.content
            else:
                # 没有工具调用，直接使用回复内容
                content = response_message.content

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
