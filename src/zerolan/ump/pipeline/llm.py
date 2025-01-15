from urllib.parse import urljoin

from zerolan.ump.common.decorator import pipeline_resolve
from zerolan.ump.abs_pipeline import AbstractPipeline
from pydantic import BaseModel
from zerolan.data.pipeline.abs_data import AbstractModelQuery
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction


class LLMPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11002"


class LLMPipeline(AbstractPipeline):

    def __init__(self, config: LLMPipelineConfig):
        super().__init__(config)
        self.predict_url = urljoin(config.server_url, '/llm/predict')
        self.stream_predict_url = urljoin(config.server_url, f'/llm/stream-predict')
        self.state_url = urljoin(config.server_url, '/llm/state')
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: LLMQuery) -> LLMPrediction | None:
        return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: AbstractModelQuery):
        return super().stream_predict(query)

    def parse_prediction(self, json_val: str) -> LLMPrediction:
        return LLMPrediction.model_validate_json(json_val)
