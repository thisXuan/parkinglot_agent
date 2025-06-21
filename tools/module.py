import base64
import datetime
import hashlib
import hmac
import json
from typing import Optional, Type

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ModuleInput(BaseModel):
    """模块查询工具的输入模式"""
    module_name: str = Field(description="要查询的模块名称")


class ModuleTool(BaseTool):
    """查询模块ID的工具"""
    name: str = "module_value"
    description: str = "用户输入模块名称，返回该模块对应的field、type和value信息"
    args_schema: Type[BaseModel] = ModuleInput

    def _build_header(self, uri: str, method: str, client_id: str, secret: str) -> dict:
        """构建请求头"""
        now_time = datetime.datetime.now(datetime.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        string_sign = method + " %s\n%s" % (uri, now_time)
        sign = hmac.new(bytes(secret, "utf-8"), bytes(string_sign, "utf-8"), hashlib.sha1).digest()
        signature = str(base64.b64encode(sign), "utf-8").replace("\n", "")
        header = {
            "Date": now_time,
            "Authorization": "MWS" + " " + client_id + ":" + signature,
            "Content-Type": "application/json;charset=UTF-8"
        }
        return header

    def _find_module_ids(self, data, target_name: str):
        """递归查找模块名称对应的ids"""
        if isinstance(data, list):
            for item in data:
                result = self._find_module_ids(item, target_name)
                if result:
                    return result
        elif isinstance(data, dict):
            # 检查当前节点的name是否匹配
            if data.get('name') == target_name:
                return data.get('ids', [])

            # 递归检查children
            if 'children' in data and data['children']:
                result = self._find_module_ids(data['children'], target_name)
                if result:
                    return result
        return None

    def _run(self, module_name: str) -> str:
        """执行模块查询"""
        try:
            uri = '/workstation/api/modules/ba'
            headers = self._build_header(uri, 'GET', 'workstation', 'aDC2Ert93raXbZDK')
            res = requests.get('https://walle.sankuai.com' + uri, headers=headers)
            json_data = json.loads(res.text)

            # 检查响应格式
            if json_data.get('ret') == 0 and 'data' in json_data:
                ids = self._find_module_ids(json_data['data'], module_name)
                if ids:
                    return f"模块名称为{module_name}，field为module，type为TERMS，value为{ids}"
                else:
                    return f"未找到模块 '{module_name}'"
            else:
                return f"API调用失败，响应: {json_data}"
        except Exception as e:
            return f"查询模块时发生错误: {str(e)}"

    async def _arun(self, module_name: str) -> str:
        return self._run(module_name)


# 创建工具实例
module_tool = ModuleTool() 