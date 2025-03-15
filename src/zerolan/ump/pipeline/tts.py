import os.path
import uuid

import requests
from pydantic import BaseModel
from zerolan.data.pipeline.tts import TTSQuery, TTSPrediction, TTSStreamPrediction
from zerolan.ump.common.utils.audio_util import check_audio_format

from zerolan.ump.abs_pipeline import CommonModelPipeline
from zerolan.ump.common.decorator import pipeline_resolve


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
        query_dict = self.parse_query(query)
        response = requests.post(url=self.urls["stream_predict_url"], stream=True,
                                 json=query_dict)
        response.raise_for_status()
        last = 0
        id = str(uuid.uuid4())
        for idx, chunk in enumerate(response.iter_content(chunk_size=1024)):
            last = idx
            yield TTSStreamPrediction(seq=idx,
                                      id=id,
                                      is_final=False,
                                      wave_data=chunk,
                                      audio_type=query.audio_type)
        yield TTSStreamPrediction(is_final=True, seq=last + 1, audio_type=query.audio_type, wave_data=b'')

    def parse_query(self, query: any) -> dict:
        return super().parse_query(query)

    def parse_prediction(self, data: any) -> TTSPrediction:
        audio_type = check_audio_format(data)
        return TTSPrediction(wave_data=data, audio_type=audio_type)
