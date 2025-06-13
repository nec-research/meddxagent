import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from .base import Model
from .utils import get_chat_messages, retry_with_exponential_backoff
from ddxdriver.logger import log


class Llama3Instruct(Model):
    USE_TRIMMING: bool = False

    def __init__(self, model_name="llama31instruct") -> None:
        load_dotenv()
        os.environ.pop("http_proxy", None)
        
        # The model is deployed with vllm, will need to setup the vllm endpoint here:
        params = {"api_key": "xxx", "base_url": "xxx"}
        self.client = OpenAI(**params)
        self.model_name = model_name

    def _trim_input(self, input_text: str, max_chars: int = 400000) -> str:
        """
        Trims input if too long. To be safe, trims to 400k chars
        """
        return input_text[:max_chars]

    @retry_with_exponential_backoff
    def __call__(
        self,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        message_history: List[Dict[str, str]] | None = None,
        max_tokens: int = None,
        temperature=0.0,
        **kwargs,
    ) -> str:

        if self.USE_TRIMMING:
            # Trim the user_prompt and system_prompt if they exceed token limits
            if user_prompt:
                user_prompt = self._trim_input(user_prompt)
            if system_prompt:
                system_prompt = self._trim_input(system_prompt)

        messages = get_chat_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            message_history=message_history,
        )
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name,
                extra_body={"stop_token_ids": [128009]},
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            raise Exception(f"Exception in calling Llama3Instruct: {e}")

        return chat_response.choices[0].message.content
