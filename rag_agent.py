import json
import logging
import time
from typing import AsyncGenerator, Dict, Any

from langchain.agents import create_react_agent, AgentExecutor
from langchain.schema import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from llms import global_retriever, decompose_query, get_contextual_question_prompt
from memory.SquirrelMemory import SquirrelMemory
from memory.squirrel_session_manager import create_session_manager
from tools.rag.embedding_oper import load_vector_store, query_vector_store
from tools.tool import tools

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = "1924642822859407379"
base_url = "https://aigc.sankuai.com/v1/openai/native"

class RAGAgent:
    def __init__(self):
        """
        初始化RAG Agent
        """
        # 使用Squirrel会话管理器
        self.session_manager = create_session_manager()

        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            streaming=True
        )

        self.tools = tools
        # 用于缓存每个会话的agent
        self._session_agents = {}

    def _get_or_create_agent_for_session(self, session_id):
        """为特定会话获取或创建Agent"""
        if session_id not in self._session_agents:
            # 创建该会话专用的memory
            memory = SquirrelMemory(self.session_manager, session_id)
            # 创建该会话专用的agent
            self._session_agents[session_id] = self._create_agent(memory=memory)

        return self._session_agents[session_id]

    def _create_agent(self, memory=None):
        """创建Agent执行器"""
        # 创建提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
你是一个强大的RAG助理, 你正在与**用户**进行结对json请求体的构建, 根据用户的问题在上下文信息中查找。

注意：
1. 必须包含'query'字段。'query'字段必须是一个数组。
2. 每个查询条件对象只能包含group、type、field、fieldType和value字段, 且必须为在文档中已经明确规定的字段。
3. 只有明确规定了"NOTERM"的字段的"type"才能设定为"NOTERM"。如果"fieldType"没有定义, 则结果不准显示"fieldType"字段。
4. 所有时间为YYYY-MM-DD HH:MM:SS 格式, 每一天的00:00:00作为开始, 23:59:59作为结束, 当前时间为: {current_time}。
5. 如果有任何不确定的信息，返回包含error字段的JSON对象，绝不编造答案。
6. 输出后**对结果进行检查**，如果多个查询条件对象的"field"和"type"相同，则合并为1个对象返回，合并"value"数组。

上下文信息：
{context_docs}

用户的问题整合为：{sub_query}
用户最新一轮的提问为: {query}

You have access to the following tools:{tools}

你的输出必须严格遵循以下格式!! Thought/Action/Action Input/Observation必须严格按照这个顺序出现!!
1. If you need to perform an action to answer the user's question, use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: [Specify the tool, choose from {tool_names}]
Action Input: [Provide the necessary input for the tool]
Observation:[Output of the tool]
(重复Thought/Action/Action Input/Observation这样的过程N轮)
Thought: Do I need to use a tool? No
Final Answer: [Provide your answer here, 必须是一个JSON对象, 包含一个名为query的数组, 其中每个元素都是一个查询条件对象。基本结构如下：
'''

2. If you can answer the user's question without performing any additional actions, use the following format:
'''
Thought: Do I need to use a tool? No
Final Answer: [Provide your answer here, 必须是一个JSON对象, 包含一个名为query的数组, 其中每个元素都是一个查询条件对象。基本结构如下：
{{'query': [{{'group': 1, 'type': 查询类型, 'field': 字段英文名, 'fieldType': 字段类型, 'value': [具体查询值]}}]}}]
'''

Begin!

Question: {sub_query}
Thought:{agent_scratchpad}
"""),
            MessagesPlaceholder(variable_name="chat_history"),
        ])

        # 创建agent
        agent = create_react_agent(
            self.llm,
            self.tools,
            prompt
        )

        # 创建agent执行器
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,  # 可选的memory参数
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        return agent_executor

    async def chat(self, message: str, session_id: str = None)-> AsyncGenerator[Dict[str, Any], None]:
        """
        与Agent对话
        """
        try:
            # 获取或创建会话
            session_id, _ = self.session_manager.get_or_create_session(session_id)

            yield {
                "event": "retrieval",
                "data": {
                    "content": "正在检索相关文档...",
                    "session_id": session_id
                }
            }

            # 获取该会话专用的agent
            agent = self._get_or_create_agent_for_session(session_id)

            retriever = global_retriever if global_retriever else load_vector_store(k=15, alpha=0.4)

            chat_history = self.session_manager.get_chat_history(session_id)
            history_rewritten_query = decompose_query(get_contextual_question_prompt(message, chat_history))

            context_docs = query_vector_store(retriever, history_rewritten_query)

            yield {
                "event": "generation",
                "data": {
                    "content": "检索完成，正在生成回答",
                    "session_id": session_id
                }
            }

            # 由于AgentExecutor可能不支持异步流式，我们使用同步方式
            # 但仍然可以通过分块返回来模拟流式效果
            full_response = ""
            try:
                response = agent.invoke({
                    "query": message,
                    "context_docs": context_docs,
                    "sub_query": history_rewritten_query,
                    "current_time": time.strftime('%Y-%m-%d')
                })
                
                full_response = response.get("output", "")
                
                # 处理中间步骤（工具调用等）
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        if len(step) >= 2:  # AgentAction, observation
                            action = step[0]
                            if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                                yield {
                                    "event": "tool_call",
                                    "data": {
                                        "tool": action.tool,
                                        "input": action.tool_input,
                                        "session_id": session_id
                                    }
                                }
                
                                 # 分块发送响应以模拟流式效果
                chunk_size = 50  # 每次发送50个字符
                for i in range(0, len(full_response), chunk_size):
                    chunk_content = full_response[i:i+chunk_size]
                    yield {
                        "event": "chunk",
                        "data": {
                            "content": chunk_content,
                            "session_id": session_id
                        }
                    }
                    # 减少延迟以避免任务堆积
                    import asyncio
                    try:
                        await asyncio.sleep(0.01)  # 减少延迟时间
                    except asyncio.CancelledError:
                        # 如果任务被取消，直接返回
                        return
                    
            except Exception as e:
                logger.error(f"Agent执行错误: {e}")
                full_response = f"抱歉，处理您的请求时出现错误: {str(e)}"

            # 发送完成信号
            yield {
                "event": "done",
                "data": {
                    "content": full_response,
                    "session_id": session_id
                }
            }

        except Exception as e:
            logger.error(f"对话处理错误: {e}")
            yield {
                "event": "done",
                "data": {
                    "content":  f"抱歉，处理您的请求时出现错误: {str(e)}",
                    "session_id": session_id
                }
            }

    def clear_session(self, session_id: str):
        """清除会话历史"""
        try:
            self.session_manager.clear_session(session_id)
            # 同时清除缓存的agent
            if session_id in self._session_agents:
                del self._session_agents[session_id]
            logger.info(f"已清除会话 {session_id} 的历史记录")
        except Exception as e:
            logger.error(f"清除会话历史错误: {e}")

    def get_session_info(self, session_id: str):
        """获取会话信息"""
        try:
            # 获取会话历史
            messages = self.session_manager.get_chat_history(session_id)

            return {
                "session_id": session_id,
                "messages": [
                    {
                        "type": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    for msg in messages
                ]
            }
        except Exception as e:
            logger.error(f"获取会话信息错误: {e}")
            return None

# 使用示例
if __name__ == "__main__":
    import asyncio

    async def main():
        agent = RAGAgent()

        session_id = None

        while True:
            try:
                user_input = input("\n用户: ")
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break

                if user_input.lower() == 'clear':
                    if session_id:
                        agent.clear_session(session_id)
                        print("会话历史已清除")
                    continue

                if user_input.lower() == 'info':
                    if session_id:
                        info = agent.get_session_info(session_id)
                        if info:
                            print(f"会话信息: {json.dumps(info, indent=2, ensure_ascii=False)}")
                    continue

                print("助手: ", end="", flush=True)
                full_response = ""

                # 异步处理流式响应
                async for chunk in agent.chat(user_input, session_id):
                    event = chunk.get("event")
                    data = chunk.get("data", {})

                    if event == "retrieval":
                        print(f"\n{data.get('content', '')}")
                        session_id = data.get('session_id', session_id)

                    elif event == "generation":
                        print(f"{data.get('content', '')}")

                    elif event == "chunk":
                        content = data.get('content', '')
                        print(content, end="", flush=True)
                        full_response += content

                    elif event == "done":
                        session_id = data.get('session_id', session_id)
                        if not full_response:  # 如果没有流式内容，显示完整响应
                            full_response = data.get('content', '')
                            print(full_response)
                        break

                print(f"\n会话ID: {session_id}")

            except KeyboardInterrupt:
                print("\n程序已退出")
                break
            except Exception as e:
                print(f"\n处理错误: {e}")
                continue

    # 运行异步主函数
    asyncio.run(main())
