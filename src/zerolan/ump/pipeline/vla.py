from pydantic import BaseModel
from zerolan.data.pipeline.vla import ShowUiQuery, ShowUiPrediction

from zerolan.ump.abs_pipeline import AbstractImagePipeline
from zerolan.ump.common.decorator import pipeline_resolve


class ShowUIConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11009"


class ShowUIPipeline(AbstractImagePipeline):

    def __init__(self, config: ShowUIConfig):
        super().__init__(config, "vla/showui")
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: ShowUiQuery) -> ShowUiPrediction | None:
        return super().predict(query)

    def stream_predict(self, query: ShowUiQuery):
        raise NotImplementedError()

    def parse_query(self, query: any) -> dict:
        return super().parse_query(query=query)

    def parse_prediction(self, json_val: any) -> ShowUiPrediction:
        return ShowUiPrediction.model_validate_json(json_val)
