import logging
from tools.rag.embedding_oper import load_vector_store
import time
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = "1924642822859407379"
base_url = "https://aigc.sankuai.com/v1/openai/native"

# 创建全局的ChatOpenAI实例
llm = ChatOpenAI(
    model="gpt-4.1", 
    api_key=api_key, 
    base_url=base_url,
    temperature=0.1
)

sub_decomposition_template = """
<system>
你是一名非常专业的AI助手，负责将复杂的筛选条件变为多个简单的子查询，以便RAG系统处理。  
请根据原始问题，将其分解为多个更简单的子查询，这些子查询的答案合起来能够全面覆盖原始问题，不改变语义，不自行添加条件。

要求:不允许遗漏原始问题中的任何条件，对于不理解的词语，不允许改写。不允许添加与原始问题无关的词语。各个子问题间不准出现重复的条件，要求子问题之间是并集关系。
</system>\n\n

原始问题：{query}
返回所有的子问题，返回格式为json:
{{
    "query": [
        "子问题1",
        "子问题2",
        "子问题3",
        "子问题4"
    ]
}}
<example>
示例：路侧-AB深圳测试，车辆型号为业务组46号电动车的case
子问题：
1. "路侧-AB深圳测试"的case
2. 车辆型号为"业务组46号电动车"的case

示例：算法能力是入口与高速的case
子问题：
1. 算法能力是入口的case
2. 算法能力是高速的case
</example>
"""

subquery_decomposition_prompt = PromptTemplate(
    input_variables=["query"],
    template=sub_decomposition_template
)

subquery_decomposer_chain = subquery_decomposition_prompt | llm

def decompose_query(query):
    return subquery_decomposer_chain.invoke({"query": query}).content

# 创建上下文问题改写的PromptTemplate
context_rewrite_template = """<system>你是一个非常专业的AI助手。请根据每一轮用户提出的问题, 改写用户最终提出的问题。你只需要改写，请不要直接回答问题。
不准改变用户的语义，不准遗漏用户的问题。\n
要求:\n
1. **准确性**: 用户的语义不能改变, 不准遗漏。\n
2. **完整性**: 每一轮用户意图的所有信息都要包括, 不准遗漏.\n
3. **关注用户新提出的问题**: 整合聊天历史，提炼出用户此时真正的意图。\n
4. 只返回改写的问题，**Never(绝不)**有其他说明文字。\n

示例:\n
历史记录: user问: "最近领取的接管", assistant回答: "查询结果..."\n
用户再次输入问题: "最近是指一个月呢"\n
返回: "最近一个月领取的接管"\n

直接返回string即可。\n
</system>

<聊天历史>
{history_text}\n
</聊天历史>
<用户新提出的问题>
{user_input}
</用户新提出的问题>
"""

context_rewrite_prompt = PromptTemplate(
    input_variables=["history_text", "user_input"],
    template=context_rewrite_template
)

context_rewriter_chain = context_rewrite_prompt | llm

def get_contextual_question_prompt(user_input, chat_history):
    '''
    基于历史记录来改写用户问的问题
    '''
    # 格式化历史记录 - 处理langchain的消息格式
    history_text = ""
    if chat_history:
        # 按顺序组织对话，确保用户输入和系统回复成对出现
        user_inputs = []
        ai_responses = []
        
        for message in chat_history:
            if isinstance(message, HumanMessage):
                user_inputs.append(str(message.content).replace('\n', ' '))
            elif isinstance(message, AIMessage):
                ai_responses.append(str(message.content).replace('\n', ' '))
        
        # 构建历史文本：用户输入1，系统回复1，用户输入2，系统回复2...
        conversation_pairs = []
        max_pairs = min(len(user_inputs), len(ai_responses))
        
        for i in range(max_pairs):
            conversation_pairs.append(f"用户输入{i+1}: {user_inputs[i]}")
            conversation_pairs.append(f"系统回复{i+1}: {ai_responses[i]}")
        
        # 如果还有未配对的用户输入
        if len(user_inputs) > max_pairs:
            for i in range(max_pairs, len(user_inputs)):
                conversation_pairs.append(f"用户输入{i+1}: {user_inputs[i]}")
        
        history_text = "，".join(conversation_pairs)
    
    if not history_text:
        history_text = "无历史记录"
    
    # 使用langchain的PromptTemplate和chain
    response = context_rewriter_chain.invoke({
        "history_text": history_text,
        "user_input": str(user_input)
    })

    return response.content

# 全局预热检索器
def warmup_retriever(k,alpha):
    try:
        logger.info("预热检索器...")
        start_time = time.time()
        
        # 提前加载检索器
        retriever = load_vector_store(k, alpha)
        logger.info(f"检索器预热完成，耗时: {time.time() - start_time:.4f}秒")
        return retriever
    except Exception as e:
        logger.error(f"检索器预热失败: {str(e)}")
        return None
    
# 应用启动时预热检索器
global_retriever = warmup_retriever(k=20, alpha=0.4)