import os.path

from pydantic import BaseModel
from zerolan.data.pipeline.tts import TTSQuery, TTSPrediction

from zerolan.ump.abs_pipeline import CommonModelPipeline
from zerolan.ump.common.decorator import pipeline_resolve
from zerolan.ump.common.utils.audio_util import check_audio_format


class TTSPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11006"


class TTSPipeline(CommonModelPipeline):

    def __init__(self, config: TTSPipelineConfig):
        super().__init__(config, "tts")
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: TTSQuery) -> TTSPrediction | None:
        if os.path.exists(query.refer_wav_path):
            query.refer_wav_path = os.path.abspath(query.refer_wav_path)
        return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: TTSQuery):
        return super().stream_predict(query)

    def parse_query(self, query: any) -> dict:
        return super().parse_query(query)

    def parse_prediction(self, data: bytes) -> TTSPrediction:
        audio_type = check_audio_format(data)
        return TTSPrediction(wave_data=data, audio_type=audio_type)
