import os
from abc import ABC, abstractmethod
from http import HTTPStatus

import requests
from loguru import logger
from pydantic import BaseModel
from zerolan.data.data.state import AppStatusEnum, ServiceState
from zerolan.data.pipeline.abs_data import AbsractImageModelQuery, AbstractModelQuery, AbstractModelPrediction

from zerolan.ump.common.utils.web_util import is_valid_url


class AbstractPipeline(ABC):

    def __init__(self, config: any):
        """
        抽象管线。
        任何管线（LLMPipeline等）都是它的子类。
        :param config: 一个用于表示该管线配置的实例。
        """
        self.config = config
        self.is_pipeline_enable()
        self.predict_url: str | None = None
        self.stream_predict_url: str | None = None
        self.state_url: str | None = None

    def is_pipeline_enable(self):
        if not self.config.enable:
            raise Exception("The pipeline is disabled in your config!")

    def check_urls(self):
        """
        用于检查配置实例中的推理 URL 是否合法。
        :return:
        """
        urls = {"predict_url": self.predict_url,
                "stream_predict_url": self.stream_predict_url,
                "state_url": self.state_url}
        for url_name, url in urls.items():
            if url is None:
                raise ValueError(f"No {url_name} URL was provided.")
            if not is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")

    @abstractmethod
    def predict(self, query: AbstractModelQuery) -> AbstractModelPrediction | None:
        """
        将一切 Query 解析为 Dict，并对目标 URL 使用 POST 请求。
        注意：该方法非异步方法，会阻塞线程。
        :param query: 对于模型的请求实例。
        :return: 返回模型的响应实例。
        """
        query_dict = self.parse_query(query)
        response = requests.post(url=self.predict_url, stream=True, json=query_dict)
        if response.status_code == HTTPStatus.OK:
            prediction = self.parse_prediction(response.content)
            return prediction
        else:
            response.raise_for_status()

    @abstractmethod
    def stream_predict(self, query: AbstractModelQuery):
        """
        将一切 Query 解析为 Dict，并对目标 URL 使用 POST 请求。
        它以 Generator 的方式迭代数据，请使用 for 循环取出其中的值。
        注意：该方法非异步方法，会阻塞线程。
        :param query: 对于模型的请求实例。
        :return: 返回模型的响应实例的 Generator。
        """
        query_dict = self.parse_query(query)
        response = requests.get(url=self.stream_predict_url, stream=True,
                                json=query_dict)

        if response.status_code == HTTPStatus.OK:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                prediction = self.parse_prediction(chunk)
                yield prediction
        else:
            response.raise_for_status()

    def parse_query(self, query: any) -> dict:
        """
        尝试将 Query 解析为 Dict，解析失败会抛出 ValueError。
        :param query: 任何继承了 BaseModel 的类实例。
        :return:
        """
        if isinstance(query, BaseModel):
            return query.model_dump()
        else:
            raise ValueError("Must be an instance of BaseModel.")

    def parse_prediction(self, json_val: any) -> AbstractModelPrediction:
        """
        尝试将 JSON 字符串或 Dict 解析为类的实例。
        :param json_val: JSON 字符串或 Dict。
        :return:
        """
        if isinstance(json_val, str):
            return AbstractModelPrediction.model_validate_json(json_val)
        elif isinstance(json_val, dict):
            return AbstractModelPrediction.model_validate_dict(json_val)
        else:
            raise ValueError("Must be an instance of BaseModel.")

    def check_state(self) -> ServiceState:
        try:
            response = requests.get(url=self.state_url, stream=True)
            if response.status_code == HTTPStatus.OK:
                state = ServiceState.model_validate_json(response.content)
                return state
        except Exception as e:
            logger.error(e)
            return ServiceState(state=AppStatusEnum.UNKNOWN, msg=f"{e}")


class AbstractImagePipeline(AbstractPipeline):
    def __init__(self, config: any):
        super().__init__(config)
        self.predict_url: str | None = None
        self.stream_predict_url: str | None = None

    def predict(self, query: AbsractImageModelQuery) -> AbstractModelPrediction | None:
        # 如果 query.img_path 的路径在本机上是存在的，那么将图片读取为二进制文件，添加到请求的 files 里
        if os.path.exists(query.img_path):
            query.img_path = os.path.abspath(query.img_path)
            files = {'image': open(query.img_path, 'rb')}

            # 将其他的字段继续序列化为 JSON 字符串
            data = {'json': query.model_dump_json()}

            response = requests.post(url=self.predict_url, files=files, data=data)
        # 如果 query.img_path 的路径在本机上是不存在的，那么认为在远程主机上一定存在
        else:
            response = requests.post(url=self.predict_url, json=query.model_dump())
        if response.status_code == HTTPStatus.OK:
            prediction = self.parse_prediction(response.content)
            return prediction
        else:
            response.raise_for_status()

    @abstractmethod
    def stream_predict(self, query: AbstractModelQuery):
        raise NotImplementedError()

    @abstractmethod
    def parse_query(self, query: any) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def parse_prediction(self, json_val: any) -> AbstractModelPrediction:
        raise NotImplementedError()
