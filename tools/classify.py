import base64
import datetime
import hashlib
import hmac
import json
from typing import Optional, Type

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ClassifyInput(BaseModel):
    """分类查询工具的输入模式"""
    classify_name: str = Field(description="要查询的分类名称")


class ClassifyTool(BaseTool):
    """查询分类ID的工具"""
    name: str = "classify_value"
    description: str = "用户输入分类名称，返回该分类对应的field、type和value信息"
    args_schema: Type[BaseModel] = ClassifyInput

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

    def _find_classify_ids(self, data, target_name: str):
        """递归查找分类名称对应的ids"""
        if isinstance(data, list):
            for item in data:
                result = self._find_classify_ids(item, target_name)
                if result:
                    return result
        elif isinstance(data, dict):
            # 检查当前节点的name是否匹配
            if data.get('name') == target_name:
                return data.get('ids', [])

            # 递归检查children
            if 'children' in data and data['children']:
                result = self._find_classify_ids(data['children'], target_name)
                if result:
                    return result
        return None

    def _run(self, classify_name: str) -> str:
        """执行分类查询"""
        try:
            uri = '/workstation/api/classifys/ba'
            headers = self._build_header(uri, 'GET', 'workstation', 'aDC2Ert93raXbZDK')
            res = requests.get('https://walle.sankuai.com' + uri, headers=headers)
            json_data = json.loads(res.text)

            # 检查响应格式
            if json_data.get('ret') == 0 and 'data' in json_data:
                ids = self._find_classify_ids(json_data['data'], classify_name)
                if ids:
                    return f"分类名称为{classify_name}，field为classify，type为TERMS，value为{ids}"
                else:
                    return f"未找到分类 '{classify_name}'"
            else:
                return f"API调用失败，响应: {json_data}"
        except Exception as e:
            return f"查询分类时发生错误: {str(e)}"

    async def _arun(self, classify_name: str) -> str:
        """异步执行分类查询"""
        # 目前使用同步实现
        return self._run(classify_name)


# 创建工具实例
classify_tool = ClassifyTool() 