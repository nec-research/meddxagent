import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import List, Dict

from .base import Model
from .utils import get_chat_messages, retry_with_exponential_backoff


class OpenAIAzureChat(Model):
    def __init__(self, model_name="gpt-4o", api_version="2024-12-01-preview") -> None:
        # For model_name, use the deployment name in Microsoft Azure rather than the openai model name
        load_dotenv()
        self.params = {
            "api_key": os.environ["OAI_KEY"],
            "azure_endpoint": os.environ["AZURE_ENDPOINT"],
            "model": model_name,
            "api_version": api_version,
        }

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
        self.params["max_tokens"] = max_tokens
        self.params["temperature"] = temperature

        self.client = AzureChatOpenAI(**self.params, **kwargs)

        messages = get_chat_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            message_history=message_history,
        )
        try:
            response = self.client.invoke(
                input=messages,
            )
        except Exception as e:
            raise Exception(f"Exception in calling OpenAIAzure: {e}")

        token_usage = response["usage"]["total_tokens"] if "usage" in response else None
        return response.content
