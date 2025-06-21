"""
工具集合模块
导入所有可用的工具
"""

from .module import module_tool
from .classify import classify_tool

# 工具列表
tools = [module_tool, classify_tool]

if __name__ == "__main__":
    result = module_tool._run("Perception")
    print(result)

    result = classify_tool._run("高精地图")
    print(result)
