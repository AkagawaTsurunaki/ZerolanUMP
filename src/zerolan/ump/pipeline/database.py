import json
from urllib.parse import urljoin

import requests
from pydantic import BaseModel
from zerolan.data.pipeline.milvus import MilvusInsert, MilvusInsertResult, MilvusQuery, MilvusQueryResult

from zerolan.ump.abs_pipeline import AbstractPipeline


class MilvusDatabaseConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11010"


def _post(url: str, obj: any, return_type: any):
    if isinstance(obj, BaseModel):
        json_val = obj.model_dump()
    else:
        json_val = obj

    response = requests.post(url=url, json=json_val)
    response.raise_for_status()

    json_val = response.json()
    if hasattr(return_type, "model_validate"):
        return return_type.model_validate(json_val)
    else:
        return json.loads(json_val)


class MilvusPipeline(AbstractPipeline):
    def __init__(self, config: MilvusDatabaseConfig):
        super().__init__(config, "milvus")
        self.urls["insert_url"] = urljoin(config.server_url, f'/{self.model_type}/insert')
        self.urls["search_url"] = urljoin(config.server_url, f'/{self.model_type}/search')
        self.check_urls()

    def insert(self, insert: MilvusInsert) -> MilvusInsertResult:
        return _post(url=self.urls["insert_url"], obj=insert, return_type=MilvusInsertResult)

    def search(self, query: MilvusQuery) -> MilvusQueryResult:
        return _post(url=self.urls["search_url"], obj=query, return_type=MilvusQueryResult)
