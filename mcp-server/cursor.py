from fastmcp import FastMCP
import requests
import json
import os
import sys
import json
import hmac
import base64
import getpass
import hashlib
import logging
from datetime import datetime as dt
import webbrowser
import time
import http.server
import socketserver
import urllib.parse
import threading

import requests

# 配置日志输出控制变量
ENABLE_LOGGING = False

logger = logging.getLogger(__name__)


def log(*args):
    print(*args, file=sys.stderr)

mcp = FastMCP("search_cases",json_response=True)

class MISTokenCache:
    def __init__(self, mis_id):
        self.token_path = "%s/.%s_sso" % (os.path.expanduser('~'), mis_id)
        self.refresh_token_path = "%s/.%s_sso_refresh" % (os.path.expanduser('~'), mis_id)

    def update(self, token, refresh_token):
        with open(self.token_path, "w") as f:
            f.write(token)
        with open(self.refresh_token_path, "w") as f:
            f.write(refresh_token)

    def get(self):
        token = None
        refresh_token = None
        if os.path.isfile(self.token_path):
            with open(self.token_path) as f:
                token = "".join(f.readlines()).strip()
        if os.path.isfile(self.refresh_token_path):
            with open(self.refresh_token_path) as f:
                refresh_token = "".join(f.readlines()).strip()
        return token, refresh_token

    def clear(self):
        if os.path.isfile(self.token_path):
            os.remove(self.token_path)
        if os.path.isfile(self.refresh_token_path):
            os.remove(self.refresh_token_path)

CIK = b'NGY5NWUzMGRmZQ=='
CSK = b'OTMxNDhkMTE4NzE4NDRlNjhlYzczNjM1MGE5ZWEyZjY='

class SSOClient:
    # FIXME: get ride of the secret from source code
    SSO_CLIENT_ID = base64.b64decode(CIK).decode('utf-8')
    SSO_CLIENT_SECRET = base64.b64decode(CSK).decode('utf-8')
    SSO_HOST = "http://sso.vip.sankuai.com"

    HOST = "walle.sankuai.com"
    API_SSO_USER = "https://" + HOST + "/api/getUserInfo"

    def __init__(self):
        self.token = None
        self.refresh_token = None

    def _build_headers(self, uri, method='POST'):
        gmt_time = dt.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
        sign_input = (method + " %s\n%s" % (uri, gmt_time)).encode("utf-8")
        signature = str(base64.encodebytes(
            hmac.new(self.SSO_CLIENT_SECRET.encode("utf-8"), sign_input, hashlib.sha1).digest()), "utf-8").strip()
        auth = ''.join(['MWS', ' ', self.SSO_CLIENT_ID, ':', signature])
        return {
            'Date': gmt_time,
            'Authorization': auth,
            'Content-Type': 'application/json'
        }

    def verify(self, token):
        headers = {
            "Accept": "application/json",
            "Cookie": "%s_ssoid=%s" % (self.SSO_CLIENT_ID, token)
        }
        res = requests.get(url=self.API_SSO_USER, headers=headers, timeout=10).json()
        return res.get("status") != 401

    def refresh(self, refresh_token):
        sso_data = {
            'refreshToken': refresh_token
        }
        uri = '/sson/oauth2.0/refresh-token'
        res = requests.post(self.SSO_HOST + uri,
                            data=json.dumps(sso_data),
                            headers=self._build_headers(uri),
                            timeout=10).json()
        if res.get("code") != 200:
            log("> 在刷新token时遇到了问题, 错误信息为:", json.dumps(res, ensure_ascii=False))
            return False
        self.token = res["data"]["accessToken"]
        self.refresh_token = res["data"]["refreshToken"]
        return True

    def login(self, mis_id, password, otp_code):
        sso_data = {
            'loginName': mis_id,
            'password': password,
            'otpCode': otp_code
        }
        uri = "/sson/api/auth"
        res = requests.post(self.SSO_HOST + uri,
                            data=json.dumps(sso_data),
                            headers=self._build_headers(uri),
                            timeout=10).json()
        if res.get("code") != 200:
            log("> 在获取token时遇到了问题, 错误信息为:", json.dumps(res, ensure_ascii=False))
            return False
        self.token = res["data"]["accessToken"]
        self.refresh_token = res["data"]["refreshToken"]
        return True

# 新增：浏览器扫码登录
class OAuthCallbackHandler(http.server.SimpleHTTPRequestHandler):
    mis_id = None

    @classmethod
    def set_mis_id(cls, mis_id):
        cls.mis_id = mis_id

    def do_GET(self):
        """处理OAuth回调"""

        # 如果不是回调URL
        if not self.path.startswith('/callback'):
            # 处理其他请求
            self.send_response(404)
            self.end_headers()

        else:
            # 解析URL中的参数（包含授权码或token）
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            # 提取授权数据
            token = None
            for key in ['token', 'access_token', 'code', 'id_token']:
                if key in params:
                    token = params[key][0]
                    break

            if token:
                # 直接保存到缓存文件
                res = requests.get('https://walle.sankuai.com/api/accessToken?code=' + token)
                token = res.json()['accessToken']
                cache = MISTokenCache(self.mis_id)
                cache.update(token, "")  # 更新token，refresh_token为空字符串

            # 返回成功页面
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()

            # 读取 HTML 文件内容
            with open('success.html', 'r', encoding='utf-8') as f:
                success_html = f.read()

            self.wfile.write(success_html.encode('utf-8'))

            # 立即关闭服务器
            threading.Thread(target=self.server.shutdown).start()


    def log_message(self, format, *args):
        """条件控制日志输出"""
        if ENABLE_LOGGING:
            super().log_message(format, *args)


def start_server(mis_id, port=8000):
    """启动本地服务器"""
    OAuthCallbackHandler.set_mis_id(mis_id)
    handler = OAuthCallbackHandler

    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            httpd.serve_forever()
    except OSError as ose:
        if ose.errno in (98, 48):
            print(f"端口 {port} 已被占用")


def browser_login(mis_id):
    """使用浏览器扫码进行登录"""

    # 启动本地服务器
    port = 8000

    server_thread = threading.Thread(target=start_server, args=(mis_id, port,))
    server_thread.daemon = True
    server_thread.start()

    # 等待服务器启动
    time.sleep(1)

    # 检查线程是否存活
    if not server_thread.is_alive():
        print(f"服务器启动失败，跳过浏览器登录，进入动态密码登录")
        return False

    # 打开浏览器，访问SSO登录页面
    sso_url = f"https://ssosv.sankuai.com/sson/login?client_id={SSOClient.SSO_CLIENT_ID}&redirect_uri=http://localhost:{port}/callback"
    webbrowser.open(sso_url)
    print("请在浏览器中完成登录，登录后将自动重定向到本地服务器。若无法自动打开浏览器，请手动访问以下链接进行登录: ")
    print(sso_url)

    # 等待服务器线程结束（当接收到回调后会自动关闭）
    server_thread.join()


def login(mis_id=None, login_type="browser"):
    if not mis_id:
        mis_id = input("请输入您的mis号(没有@meituan.com后缀): ")

    def ret_value(token):
        return {
            'SSOClientId': SSOClient.SSO_CLIENT_ID,
            'token': token,
        }

    client = SSOClient()
    cache = MISTokenCache(mis_id)
    token, refresh_token = cache.get()

    # 先尝试使用缓存中的 token
    if token and client.verify(token):
        return ret_value(token)
    if refresh_token and client.refresh(refresh_token):
        cache.update(client.token, client.refresh_token)
        return ret_value(client.token)

    # 浏览器扫码登录尝试
    try:
        print("浏览器扫码登录尝试...")
        browser_login(mis_id)  # 尝试浏览器扫码登录

        token, refresh_token = cache.get()
        if token and client.verify(token):
            print("浏览器扫码登录成功！")
            return ret_value(client.token)
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
    except ValueError as ve:
        print(f"验证相关错误: {ve}")
    except Exception as e:  # 捕获其他特定异常
        print(f"浏览器扫码登录过程中发生错误: {e}")

    # 如果浏览器登录失败，使用动态密码登录
    print("尝试使用动态密码登录...")
    password = getpass.getpass("请输入您的密码:")
    otp_code = input("> 请输入大象动态验证码: ")

    if client.login(mis_id, password, otp_code):
        cache.update(client.token, client.refresh_token)
        return ret_value(client.token)

def workstation_ai(query: str) -> str:
    '''
    将输入的查询(query)转换为case3.0工作台ai查询的请求体，返回{raw_json_responses}数据。
    '''
    try:
        url = "https://general-faas.vip.sankuai.com/api/workstation/agent"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json,text/event-stream",
            "Cache-Control": "no-cache"
        }

        payload = {"query": query}

        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)

        if response.status_code != 200:
            return f"请求失败，状态码: {response.status_code}, 响应: {response.text}"

        raw_json_responses = []
        current_event = None

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("event: "):
                current_event = line[7:].strip()
                continue

            if line.startswith("data: "):
                data_content = line[6:].strip()

                try:
                    data_json = json.loads(data_content)
                    
                    if current_event:
                        data_json["event_type"] = current_event
                    
                    raw_json_responses.append(data_json)

                    if current_event == "error" or "error" in data_json:
                        return json.dumps(raw_json_responses, ensure_ascii=False, indent=2)

                    if current_event == "done" or current_event == "completed":
                        return json.dumps(raw_json_responses, ensure_ascii=False, indent=2)

                except json.JSONDecodeError:
                    # 处理非JSON数据
                    raw_json_responses.append({
                        "content": data_content,
                        "event_type": current_event or "chunk"
                    })

        return json.dumps(raw_json_responses, ensure_ascii=False, indent=2) if raw_json_responses else "未能获取到有效响应"

    except requests.exceptions.Timeout:
        return json.dumps({"error": "请求超时，请稍后重试"}, ensure_ascii=False)
    except requests.exceptions.ConnectionError:
        return json.dumps({"error": "连接错误，请检查网络连接"}, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"请求异常: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"发生未知错误: {str(e)}"}, ensure_ascii=False)

@mcp.tool()
def search_cases(query: str, mis_id: str = None) -> str:
    '''
    使用自然语言查询搜索案例。首先将查询转换为请求体，然后使用SSO token调用搜索API。
    
    参数:
    - query: 自然语言查询，如"最近创建的case"、"昨天领取的接管"等
    - mis_id: MIS账号，如果不提供会尝试从环境变量或提示输入
    '''
    try:
        if not mis_id:
            mis_id = os.environ.get('MIS_ID')
            if not mis_id:
                mis_id = input("请输入您的mis号(没有@meituan.com后缀): ")
        
        auth_result = login(mis_id)
        if not auth_result or not auth_result.get('token'):
            return json.dumps({"error": "SSO登录失败，无法获取token"}, ensure_ascii=False)
        
        token = auth_result['token']
        client_id = auth_result['SSOClientId']

        query_result = workstation_ai(query)

        try:
            query_data = json.loads(query_result)
            # 查找包含最终查询结果的项
            final_query = None
            for item in query_data:
                if item.get("event_type") == "done" and "content" in item:
                    try:
                        final_query = json.loads(item["content"])
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not final_query:
                # 如果没有找到done事件，尝试查找最后一个包含完整JSON的内容
                for item in reversed(query_data):
                    if "content" in item:
                        try:
                            final_query = json.loads(item["content"])
                            if "query" in final_query:
                                break
                        except json.JSONDecodeError:
                            continue
            
            if not final_query or "query" not in final_query:
                return json.dumps({"error": "无法解析查询条件", "raw_response": query_result}, ensure_ascii=False)
                
        except json.JSONDecodeError:
            return json.dumps({"error": "查询条件转换失败", "raw_response": query_result}, ensure_ascii=False)
        
        search_url = 'https://walle.sankuai.com/workstation/api/case/search'
        
        cookies = {
            f'{client_id}_ssoid': token,
        }
        
        response = requests.post(search_url, cookies=cookies, json=final_query, timeout=30)
        
        if response.status_code == 200:
            result_data = response.json()
            return json.dumps(result_data, ensure_ascii=False)
        else:
            return json.dumps({
                "error": f"搜索API请求失败: {response.status_code}",
                "response_text": response.text,
                "query_used": final_query
            }, ensure_ascii=False)
            
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"网络请求异常: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"发生未知错误: {e}"}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=5000)