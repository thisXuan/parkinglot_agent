from flask import Flask, Response, stream_with_context, request, jsonify
import logging
from flask_cors import CORS
import json
import time
from rag_agent import RAGAgent

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Agent = None
def get_rag_agent():
    """获取RAGAgent实例（懒加载单例）"""
    global Agent
    if Agent is None:
        logger.info("初始化RAGAgent实例...")
        Agent = RAGAgent()
        logger.info("RAGAgent实例初始化完成")
    return Agent

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

# '/api/python/demo/' 需要和Nest HTTP触发器对应
@app.route('/api/workstation/agent', methods=['POST'])
def echo():
    async def generate():
        try:
            message_id = 1
            
            # 获取POST请求的JSON数据
            data = request.get_json(silent=True)
            if not data:
                error_msg = {"error": "请求体不能为空"}
                yield format_sse(error_msg, event="error", id=message_id)
                return
                
            # 从请求体中获取查询、会话ID
            query = data.get('query')
            session_id = data.get('session_id')  # 新增：获取会话ID

            if not query:
                error_msg = {"error": "缺少query参数"}
                yield format_sse(error_msg, event="error", id=message_id)
                return

            agent = get_rag_agent()

            accumulated_text = ""
            async for chunk in agent.chat(query, session_id):
                event = chunk.get("event")
                data_content = chunk.get("data", {})
                
                if event == "retrieval":
                    # 检索阶段
                    yield format_sse({
                        "content": data_content.get('content', ''),
                        "session_id": data_content.get('session_id', session_id)
                    }, event="retrieval", id=message_id)
                    session_id = data_content.get('session_id', session_id)
                    message_id += 1
                
                elif event == "generation":
                    # 生成阶段
                    yield format_sse({
                        "content": data_content.get('content', ''),
                        "session_id": data_content.get('session_id', session_id)
                    }, event="generation", id=message_id)
                    message_id += 1
                
                elif event == "chunk":
                    # 流式内容块
                    content = data_content.get('content', '')
                    accumulated_text += content
                    yield format_sse({
                        "content": content,
                        "session_id": data_content.get('session_id', session_id)
                    }, event="chunk", id=message_id)
                    message_id += 1
                
                elif event == "tool_call":
                    # 工具调用
                    yield format_sse({
                        "tool": chunk.get('tool', ''),
                        "input": chunk.get('input', ''),
                        "session_id": chunk.get('session_id', session_id)
                    }, event="tool_call", id=message_id)
                    message_id += 1
                
                elif event == "done":
                    final_content = accumulated_text or data_content.get('content', '')
                    yield format_sse({
                        "content": final_content,
                        "status": "complete",
                        "session_id": data_content.get('session_id', session_id)
                    }, event="done", id=message_id)
                    break

        except Exception as e:
            logger.error(f"处理请求时发生错误: {str(e)}")
            yield format_sse({"error": str(e)}, event="error", id=message_id)

    # 创建同步包装器来处理异步生成器
    def sync_generate():
        import asyncio
        
        # 尝试获取现有的事件循环，如果没有则创建新的
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async_gen = None
        try:
            async_gen = generate()
            while True:
                try:
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
        except Exception as e:
            logger.error(f"同步包装器错误: {e}")
            yield format_sse({"error": str(e)}, event="error", id=1)
        finally:
            # 确保异步生成器被正确关闭
            if async_gen:
                try:
                    loop.run_until_complete(async_gen.aclose())
                except Exception as e:
                    logger.error(f"关闭异步生成器错误: {e}")

    return Response(stream_with_context(sync_generate()),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'  # 禁用Nginx缓冲
                   })

# 获取会话历史的API端点
@app.route('/api/workstation/agent/history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """获取特定会话的历史记录"""
    try:
        return Agent.get_session_info(session_id)
    except Exception as e:
        logger.error(f"获取会话历史失败: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# 清空会话历史的API端点
@app.route('/api/workstation/agent/clear/<session_id>', methods=['POST'])
def clear_session_history(session_id):
    """清空特定会话的历史记录"""
    try:
        Agent.clear_session(session_id)
        return jsonify({
            "session_id": session_id,
            "message": "会话历史已清空"
        })
    except Exception as e:
        logger.error(f"清空会话历史失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

# wsgi标准入口，文件nest-handler.py 必须提供一个形如如下格式的handler。
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)