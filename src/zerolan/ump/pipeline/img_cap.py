from urllib.parse import urljoin

from pydantic import BaseModel
from zerolan.data.pipeline.abs_data import AbstractModelQuery
from zerolan.data.pipeline.img_cap import ImgCapQuery, ImgCapPrediction

from zerolan.ump.abs_pipeline import AbstractImagePipeline
from zerolan.ump.common.decorator import pipeline_resolve


class ImgCapPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11003"


class ImgCapPipeline(AbstractImagePipeline):

    def __init__(self, config: ImgCapPipelineConfig):
        super().__init__(config)
        self.predict_url = urljoin(config.server_url, '/img-cap/predict')
        self.stream_predict_url = urljoin(config.server_url, '/img-cap/stream-predict')
        self.state_url = urljoin(config.server_url, '/img-cap/state')
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: ImgCapQuery) -> ImgCapPrediction | None:
        return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: AbstractModelQuery):
        raise NotImplementedError()

    def parse_query(self, query: any) -> dict:
        return super().parse_query(query)

    def parse_prediction(self, json_val: str) -> ImgCapPrediction:
        return ImgCapPrediction.model_validate_json(json_val)
