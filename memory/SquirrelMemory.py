from langchain.memory.chat_memory import BaseChatMemory
from typing import Dict, List, Any

class SquirrelMemory(BaseChatMemory):
    """基于Squirrel的聊天记忆实现"""
    
    def __init__(self, session_manager, session_id, **kwargs):
        # 先调用父类初始化
        super().__init__(**kwargs)
        # 使用对象属性而不是字段
        object.__setattr__(self, 'session_manager', session_manager)
        object.__setattr__(self, 'session_id', session_id)
        object.__setattr__(self, 'memory_key', "chat_history")

    @property
    def memory_variables(self) -> List[str]:
        """返回内存变量列表"""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载内存变量"""
        messages = self.session_manager.get_chat_history(self.session_id)
        return {self.memory_key: messages}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存对话上下文"""
        input_str = inputs.get("input", "")
        output_str = outputs.get("output", "")
        self.session_manager.add_message(self.session_id, input_str, output_str)

    def clear(self) -> None:
        """清空内存"""
        self.session_manager.clear_session(self.session_id)