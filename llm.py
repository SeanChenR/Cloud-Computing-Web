"""Wrapper LLM conversation APIs."""
import os
import json
import requests
import configparser

from pydantic.v1 import BaseModel, Field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.messages.chat import ChatMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from langchain.memory.summary_buffer import ConversationSummaryBufferMemory

from langchain_core.callbacks.manager import (
    CallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain

class _ChatFormosaFoundationCommon(BaseLanguageModel):
    base_url: str = "http://localhost:12345"
    """Base url the model is hosted under."""

    model: str = "ffm-mixtral-8x7b-32k-instruct"
    """Model name to use."""
    
    temperature: Optional[float]
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively."""

    stop: Optional[List[str]]
    """Sets the stop tokens to use."""

    top_k: int = 50
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 50)"""

    top_p: float = 1
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 1)"""

    max_new_tokens: int = 350
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

    frequence_penalty: float = 1
    """Penalizes repeated tokens according to frequency."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    ffm_api_key: Optional[str] = None
        
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling FFM API."""
        normal_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "frequence_penalty": self.frequence_penalty,
            "top_k": self.top_k,
        }
        return {**normal_params, **self.model_kwargs}

    def _call(
        self,
        prompt,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []
        params = {**self._default_params, "stop": stop, **kwargs}
        parameter_payload = {"parameters": params, "messages": prompt, "model": self.model}

        # HTTP headers for authorization
        headers = {
            "X-API-KEY": self.ffm_api_key,
            "X-API-HOST": "afs-inference",
            "Content-Type": "application/json",
        }
        endpoint_url = f"{self.base_url}/models/conversation"
        # send request
        try:
            response = requests.post(
                url=endpoint_url, 
                headers=headers, 
                data=json.dumps(parameter_payload, ensure_ascii=False).encode("utf8"),
                stream=False,
            )
            response.encoding = "utf-8"
            generated_text = response.json()
            if response.status_code != 200:
                detail = generated_text.get("detail")
                raise ValueError(
                    f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                    f"error raised with status code {response.status_code}\n"
                    f"Details: {detail}\n"
                )

        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(f"FormosaFoundationModel error raised by inference endpoint: {e}\n")

        if generated_text.get("detail") is not None:
            detail = generated_text["detail"]
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f"error raised by inference API: {detail}\n"
            )
        
        if generated_text.get("generated_text") is None:
            raise ValueError(
                f"FormosaFoundationModel endpoint_url: {endpoint_url}\n"
                f"Response format error: {generated_text}\n"
            )

        return generated_text

class ChatFormosaFoundationModel(BaseChatModel, _ChatFormosaFoundationCommon):
    """`FormosaFoundation` Chat large language models API.

    The environment variable ``OPENAI_API_KEY`` set with your API key.

    Example:
        .. code-block:: python
            ffm = ChatFormosaFoundationModel(model_name="llama2-7b-chat-meta")
    """

    @property
    def _llm_type(self) -> str:
        return "ChatFormosaFoundationModel"

    @property
    def lc_serializable(self) -> bool:
        return True

    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "human", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_dict

    def _create_conversation_messages(
        self,
        messages: List[BaseMessage], 
        stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params: Dict[str, Any] = {**self._default_params}

        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        chat_generation = ChatGeneration(
            message = AIMessage(content=response.get("generated_text")),
            generation_info = {
                "token_usage": response.get("generated_tokens"), 
                "model": self.model
            }
        )

        return ChatResult(generations=[chat_generation])

    def _generate(
        self,         
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self._call(prompt=message_dicts)
        if type(response) is str: # response is not the format of dictionary
            return response

        return self._create_chat_result(response)

    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> ChatResult:
        pass
    
    def _create_message_dicts(
        self, 
        messages: List[BaseMessage], 
        stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [self._convert_message_to_dict(m) for m in messages]
        return message_dicts, params
    

config = configparser.ConfigParser()
config.read('config.ini')

MODEL_NAME = "ffm-mixtral-8x7b-32k-instruct"
API_KEY = config['llama3']['API_KEY']
API_URL = config['llama3']['API_URL']

chat_ffm = ChatFormosaFoundationModel(
        base_url = API_URL,
        max_new_tokens = 350,
        temperature = 0.5,
        top_k = 50,
        top_p = 1.0,
        frequence_penalty = 1.0,
        ffm_api_key = API_KEY,
        model = MODEL_NAME
    )

memory = ConversationSummaryBufferMemory(
    llm=chat_ffm, max_token_limit=200, return_messages=True
)

# from langchain.memory.buffer import ConversationBufferMemory
# from langchain.chains import ConversationChain

# prompt_template = """你是一個籃球員，請都使用專業球評的語氣回答以下問題，Q：{input}
#                     對話紀錄：{history}"""
# prompt = PromptTemplate(input_variables=["input", "history"], template=prompt_template)

# # llm = prompt | chat_ffm | chat_memory
# # result = llm.invoke({"man":"籃球迷", "say":"專業球評", "question":"NBA歷史上得分最多的球員是誰，請用繁體中文回答我，如果是外國人，名字請以英文表示"})
# # print(result.content)

# conversation = ConversationChain(
#     prompt=prompt,
#     llm=chat_ffm,
#     verbose=True,
#     memory=ConversationBufferMemory(),
# )

# ans = conversation.predict(input="NBA歷史上得分最多的球員是誰，請用繁體中文回答我，如果是外國人，名字請以英文表示")
# print(ans)