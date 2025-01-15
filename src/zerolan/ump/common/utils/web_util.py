import re


def is_valid_url(url: str | None) -> bool:
    protocol = url.split("://")[0]
    return protocol in ['http', 'https']


def is_html_string(s):
    pattern = r'<(html|head|body|div|span|p|a|img|table|tr|td|th|ul|ol|li|form|input|textarea|select|option)[^>]*>'
    return bool(re.search(pattern, s))
