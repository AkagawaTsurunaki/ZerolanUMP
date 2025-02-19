from openai import OpenAI
from pydantic import BaseModel
from zerolan.data.pipeline.abs_data import AbstractModelQuery
from zerolan.data.pipeline.llm import LLMQuery, LLMPrediction, RoleEnum, Conversation

from zerolan.ump.abs_pipeline import CommonModelPipeline
from zerolan.ump.common.decorator import pipeline_resolve


def _to_openai_format(query: LLMQuery):
    messages = []
    for chat in query.history:
        messages.append({
            "role": chat.role,
            "content": chat.content
        })
    messages.append({
        "role": "user",
        "content": query.text
    })
    return messages


class LLMPipelineConfig(BaseModel):
    enable: bool = True
    model: str | None = None  # ["moonshot-v1-8k", "deepseek-chat"]
    api_key: str | None = None
    server_url: str = "http://127.0.0.1:11002"


def _openai_predict(query: LLMQuery, wrapper):
    messages = _to_openai_format(query)
    completion = wrapper(messages)
    resp = completion.choices[0].message.content
    query.history.append(Conversation(role=RoleEnum.user, content=query.text))
    query.history.append(Conversation(role=RoleEnum.assistant, content=resp))
    return LLMPrediction(response=resp, history=query.history)


class LLMPipeline(CommonModelPipeline):

    def __init__(self, config: LLMPipelineConfig):
        super().__init__(config, "llm")
        self._model = config.model
        # Kimi API supported
        # Reference: https://platform.moonshot.cn/docs/guide/start-using-kimi-api
        # Deepseek API supported
        # Reference: https://api-docs.deepseek.com/zh-cn/
        if self._model in ["moonshot-v1-8k", "deepseek-chat"]:
            self._remote_model = OpenAI(api_key=config.api_key, base_url=config.server_url)
            self._remote_api = True
        else:
            self.check_urls()
            self._remote_api = False

    @pipeline_resolve()
    def predict(self, query: LLMQuery) -> LLMPrediction | None:
        if self._remote_api:
            if self._model == "moonshot-v1-8k":
                def wrapper_kimi(messages):
                    return self._remote_model.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=0.3
                    )

                return _openai_predict(query, wrapper_kimi)
            elif self._model == "deepseek-chat":
                def wrapper_deepseek(messages):
                    return self._remote_model.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        stream=False
                    )

                return _openai_predict(query, wrapper_deepseek)
            else:
                raise NotImplementedError(f"Unsupported model {self._model}")
        else:
            return super().predict(query)

    @pipeline_resolve()
    def stream_predict(self, query: AbstractModelQuery):
        return super().stream_predict(query)

    def parse_prediction(self, json_val: str) -> LLMPrediction:
        return LLMPrediction.model_validate_json(json_val)
