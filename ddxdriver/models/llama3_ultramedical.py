import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from .base import Model
from .utils import retry_with_exponential_backoff, get_condensed_chat_messages
from ddxdriver.logger import log

class Llama370BUltraMedical(Model):
    def __init__(self, model_name="Llama-3-70B-UltraMedical", num_workers: int = 16) -> None:
        load_dotenv()
        os.environ.pop("http_proxy", None)
        
        # The model is deployed with vllm, need to setup the vllm endpoint here:
        params = {
            "api_key": "xxx",
            "base_url": "xxx"
        }
        self.client = OpenAI(**params)
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _trim_input(
        self,
        input_text: str | None = None,
        system_prompt: str | None = None,
        message_history: List[Dict[str, str]] | None = None,
        max_chars: int = 28000
    ) -> tuple[str | None, List[Dict[str, str]] | None]:
        """Trims either the first user message in history or the input text if no history.
        Uses 28k chars as safe limit.
        
        Returns:
            Tuple of (trimmed_input_text, trimmed_message_history)
        """
        if message_history:
            # Make a copy to avoid modifying the original
            message_history = [msg.copy() for msg in message_history]
            
            # Find first user message and trim it
            for msg in message_history:
                if msg["role"] == "user":
                    msg["content"] = msg["content"][:max_chars]
                    break
        elif input_text:
            # If no message history, trim the input text
            input_text = input_text[:max_chars]
        
        return input_text, message_history

    def _process_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            
            # Clean up the response by removing the special tokens
            response_text = chat_response.choices[0].message.content.replace(
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>", ""
            )
            return response_text
        except Exception as e:
            raise Exception(f"Exception in calling Llama370BUltraMedical: {e}")

    @retry_with_exponential_backoff
    def __call__(
        self,
        user_prompt: str | None = None,
        system_prompt: str | None = None,
        message_history: List[Dict[str, str]] | None = None,
        max_tokens: int = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        # Trim both input and message history
        user_prompt, message_history = self._trim_input(
            user_prompt,
            system_prompt,
            message_history
        )

        messages = get_condensed_chat_messages(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            message_history=message_history,
        )

        future = self.executor.submit(
            self._process_request,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return future.result() 