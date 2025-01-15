from http import HTTPStatus
from typing import Literal

import requests
from loguru import logger
from pydantic import BaseModel
from zerolan.data.pipeline.asr import ASRQuery, ASRPrediction, ASRStreamQuery

from src.zerolan.ump.abs_pipeline import CommonModelPipeline
from zerolan.ump.common.decorator import pipeline_resolve


class ASRPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11001"
    sample_rate: int = 16000
    channels: int = 1
    format: Literal["float32"] = "float32"


class ASRPipeline(CommonModelPipeline):

    def __init__(self, config: ASRPipelineConfig):
        super().__init__(config, model_type="asr")
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: ASRQuery) -> ASRPrediction | None:
        assert isinstance(query, ASRQuery)
        try:
            files, data = self.parse_query(query)
            response = requests.post(url=self.urls["predict_url"], files=files, data=data)

            if response.status_code == HTTPStatus.OK:
                prediction = self.parse_prediction(response.content)
                return prediction

        except Exception as e:
            logger.exception(e)
            return None

    @pipeline_resolve()
    def stream_predict(self, query: ASRStreamQuery):
        files, data = self.parse_query(query)
        response = requests.get(url=self.urls["stream_predict_url"], files=files, data=data)

        if response.status_code == HTTPStatus.OK:
            return self.parse_prediction(response.content)
        else:
            response.raise_for_status()

    def parse_query(self, query: ASRQuery | ASRStreamQuery) -> tuple:
        if isinstance(query, ASRQuery):
            files = {"audio": open(query.audio_path, 'rb')}
            data = {"json": query.model_dump_json()}

            return files, data
        elif isinstance(query, ASRStreamQuery):
            files = {"audio": query.audio_data}
            query.audio_data = ""
            data = {"json": query.model_dump_json()}

            return files, data
        else:
            raise ValueError("Can not convert query.")

    def parse_prediction(self, json_val: str) -> ASRPrediction:
        return ASRPrediction.model_validate_json(json_val)
