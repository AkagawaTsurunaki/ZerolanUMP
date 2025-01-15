import os

from pydantic import BaseModel
from zerolan.data.pipeline.abs_data import AbstractModelQuery
from zerolan.data.pipeline.vid_cap import VidCapQuery, VidCapPrediction

from zerolan.ump.abs_pipeline import CommonModelPipeline
from zerolan.ump.common.decorator import pipeline_resolve


class VidCapPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11005"


class VidCapPipeline(CommonModelPipeline):

    def __init__(self, config: VidCapPipelineConfig):
        """
        此接口保留，但是可能会在将来废弃而放弃维护
        :param config:
        """
        super().__init__(config, "vid-cap")
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: VidCapQuery) -> VidCapPrediction | None:
        assert os.path.exists(query.vid_path), f"视频路径不存在：{query.vid_path}"
        return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: AbstractModelQuery):
        raise NotImplementedError()

    def parse_prediction(self, json_val: any) -> VidCapPrediction:
        return VidCapPrediction.model_validate(json_val)
