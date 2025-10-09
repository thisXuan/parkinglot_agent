import logging
import json
import time
import os
import uuid
from typing import List, Dict, Any, Optional
from squirrel.squirrelClient import SquirrelClient
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

from pycat import Cat
Cat.init_cat("com.sankuai.walledata.workstation.ai", disable_falcon=True)

class SquirrelSessionManager:
    def __init__(self):
        # Squirrel客户端配置
        cluster_name = "redis-mad-public_dev"
        _root_path = os.path.dirname(os.path.abspath(__file__))
        proxy_path = _root_path + "/squirrel-proxy.conf"
        print(proxy_path)

        self.client = SquirrelClient(
            proxy_config_path=proxy_path,
            cluster_name=cluster_name,
            print_debug_log=True,
            restart_t_healthy=True,
            appkey="com.sankuai.walledata.workstation.ai"
        )
        
        self.category = "workstation-case-agent"
        self.ttl = int(os.getenv('SESSION_TTL', 1800))  # 30分钟
        logger.info("SquirrelSessionManager 初始化完成")

    def _get_real_key(self, session_id: str) -> str:
        """生成真实的Redis key"""
        return self.client.get_category_key(self.category, session_id)

    def _serialize_message(self, message) -> str:
        """序列化消息为JSON字符串"""
        if isinstance(message, HumanMessage):
            return json.dumps({
                'type': 'human',
                'content': message.content,
                'timestamp': time.time()
            }, ensure_ascii=False)
        elif isinstance(message, AIMessage):
            return json.dumps({
                'type': 'ai', 
                'content': message.content,
                'timestamp': time.time()
            }, ensure_ascii=False)
        return json.dumps(message, ensure_ascii=False)
    
    def _deserialize_message(self, data: str):
        """反序列化JSON字符串为消息对象"""
        try:
            msg_data = json.loads(data)
            if msg_data.get('type') == 'human':
                return HumanMessage(content=msg_data['content'])
            elif msg_data.get('type') == 'ai':
                return AIMessage(content=msg_data['content'])
            return msg_data
        except Exception as e:
            logger.warning(f"反序列化消息失败: {e}")
            return None
    
    def get_or_create_session(self, session_id: Optional[str] = None):
        """获取或创建会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())

        real_key = self._get_real_key(session_id)

        # 检查会话是否存在
        try:
            length = self.client.llen(real_key)
            exists = length > 0
        except:
            exists = False
        
        if not exists:
            # 创建空会话并设置过期时间
            try:
                self.client.lpush(real_key, "")
                self.client.lpop(real_key)
                self.client.expire(real_key, self.ttl)
            except Exception as e:
                logger.error(f"创建会话失败: {e}")
        else:
            # 更新过期时间
            self.client.expire(real_key, self.ttl)
        
        # 返回兼容格式
        session_data = {
            'memory': None,
            'chat_history': self._get_chat_proxy(session_id),
            'created_at': time.time()
        }
        
        return session_id, session_data
    
    def _get_chat_proxy(self, session_id: str):
        """获取聊天历史代理对象"""
        class ChatProxy:
            def __init__(self, manager, session_id):
                self.manager = manager
                self.session_id = session_id
            
            @property
            def messages(self):
                return self.manager.get_chat_history(self.session_id)
            
            def add_user_message(self, message: str):
                self.manager._add_message(self.session_id, HumanMessage(content=message))
            
            def add_ai_message(self, message: str):
                self.manager._add_message(self.session_id, AIMessage(content=message))
            
            def clear(self):
                self.manager.clear_session(self.session_id)
        
        return ChatProxy(self, session_id)
    
    def _add_message(self, session_id: str, message):
        """添加单个消息"""
        try:
            real_key = self._get_real_key(session_id)
            serialized = self._serialize_message(message)
            
            # 添加到列表末尾
            self.client.rpush(real_key, serialized)
            # 更新过期时间
            self.client.expire(real_key, self.ttl)
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            raise
    
    def add_message(self, session_id: str, user_message: str, ai_message: str):
        """添加用户消息和AI回复"""
        self._add_message(session_id, HumanMessage(content=user_message))
        self._add_message(session_id, AIMessage(content=ai_message))
    
    def get_chat_history(self, session_id: str) -> List:
        """获取会话历史"""
        try:
            real_key = self._get_real_key(session_id)
            logger.info(f"获取会话历史: {real_key}")
            
            # 获取所有消息
            raw_messages = self.client.lrange(real_key, 0, -1)
            if not raw_messages:
                return []
            
            messages = []
            for raw_msg in raw_messages:
                msg = self._deserialize_message(raw_msg)
                if msg:
                    messages.append(msg)
            
            # 更新过期时间
            if messages:
                self.client.expire(real_key, self.ttl)
            
            return messages
            
        except Exception as e:
            logger.error(f"获取历史失败: {e}")
            return []
    
    def get_session_ttl(self, session_id: str) -> int:
        """获取会话剩余过期时间（秒）"""
        try:
            real_key = self._get_real_key(session_id)
            ttl = self.client.ttl(real_key)
            logger.info(f"会话 {session_id} 剩余时间: {ttl} 秒")
            return ttl if ttl is not None else -1
        except Exception as e:
            logger.error(f"获取会话TTL失败: {e}")
            return -1
    
    def clear_session(self, session_id: str):
        """清空会话"""
        try:
            real_key = self._get_real_key(session_id)
            self.client.delete(real_key)
            logger.info(f"清空会话: {session_id}")
        except Exception as e:
            logger.error(f"清空会话失败: {e}")
            raise

# 简单的工厂函数
def create_session_manager():
    """创建会话管理器"""
    use_squirrel = os.getenv('USE_SQUIRREL_SESSION', 'true').lower() == 'true'
    
    if use_squirrel:
        try:
            return SquirrelSessionManager()
        except Exception as e:
            logger.warning(f"Squirrel初始化失败: {e}")
            raise
    else:
        raise