from typing import List

from pydantic import BaseModel
from zerolan.data.pipeline.abs_data import AbstractModelQuery
from zerolan.data.pipeline.ocr import OCRQuery, OCRPrediction, RegionResult

from zerolan.ump.abs_pipeline import AbstractImagePipeline
from zerolan.ump.common.decorator import pipeline_resolve


class OCRPipelineConfig(BaseModel):
    enable: bool = True
    server_url: str = "http://127.0.0.1:11004"


class OCRPipeline(AbstractImagePipeline):

    def __init__(self, config: OCRPipelineConfig):
        super().__init__(config, "ocr")
        self.check_urls()

    @pipeline_resolve()
    def predict(self, query: OCRQuery) -> OCRPrediction | None:
        return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: AbstractModelQuery):
        raise NotImplementedError()

    def parse_query(self, query: any) -> dict:
        return super().parse_query(query)

    def parse_prediction(self, json_val: str) -> OCRPrediction:
        return OCRPrediction.model_validate_json(json_val)


def avg_confidence(p: OCRPrediction) -> float:
    results = len(p.region_results)
    if results == 0:
        return 0
    confidence_sum = 0
    for region_result in p.region_results:
        confidence_sum += region_result.confidence
    return confidence_sum / results


def stringify(region_results: List[RegionResult]):
    assert isinstance(region_results, list)
    for region_result in region_results:
        assert isinstance(region_result, RegionResult)

    result = ""
    for i, region_result in enumerate(region_results):
        line = f"[{i}] {region_result.content} \n"
        result += line
    return result
