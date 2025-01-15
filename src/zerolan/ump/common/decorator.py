import json
from functools import wraps

from loguru import logger
from pydantic import ValidationError
from requests import HTTPError

from zerolan.ump.common.utils.web_util import is_html_string

json_decode_err_msg = """
管线解码 JSON 时遇到异常。你可以尝试：
1. 检查你与 Zerolan Core 服务器的连接。
2. 你是否被你的服务器拦截了？请把请求 URL 复制到浏览器里访问试一次。
3. 提交 Issue。
"""

conn_err_msg = """
管线与 Zerolan Core 服务器的连接发生异常。你可以尝试：
1. 检查你与 Zerolan Core 服务器的连接。
2. 检查你的管线配置中，关于 URL 的书写是否正确？
"""

http_err_500_msg = """
500 内部服务器错误，请不要惊慌，这是很常见的错误！
你可以在 Issue 里描述该问题。
"""

http_err_404_msg = """
404 是一个经典的错误，我推测是你的 URL 出现了问题！
你可以在 Issue 里描述该问题。
"""

html_content_err_msg = """
返回的内容不是 JSON 字符串，而可能是一个 HTML 网页？！
1. 请检查你的服务器是否被权限校验、防火墙或其他组件拦截。
2. 检查你的 Nginx 配置是否正确。
"""


def pipeline_resolve():
    """
    为用户提供可能的报错解决方案的装饰器，只读取异常而不会拦截。
    :return:
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                ret = func(*args, **kwargs)
                return ret
            except Exception as e:
                if isinstance(e, json.decoder.JSONDecodeError):
                    logger.error(json_decode_err_msg)
                elif isinstance(e, ValidationError):
                    if is_html_string(e.json()):
                        logger.error(html_content_err_msg)
                elif isinstance(e, ConnectionError):
                    logger.error(conn_err_msg)
                elif isinstance(e, HTTPError):
                    if e.response.status_code == 404:
                        logger.error(http_err_500_msg)
                    if e.response.status_code == 500:
                        logger.error(http_err_500_msg)
                raise e

        return wrapper

    return decorator
