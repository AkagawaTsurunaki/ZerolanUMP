def is_valid_url(url: str | None) -> bool:
    protocol = url.split("://")[0]
    return protocol in ['http', 'https']
