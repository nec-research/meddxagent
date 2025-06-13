from typing import List, Dict
import torch
import transformers
from .base import Model
from .utils import get_single_user_prompt
from ddxdriver.logger import log
import os
import gc
MAX_NEW_TOKENS = 4096

# Global pipeline storage
_GLOBAL_PIPELINE = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class Llama318B(Model):
    def __init__(self, model_name: str) -> None:
        """Initialize the model with either Meta or UltraMedical model.
        
        Args:
            model_name: Either "meta-llama/Meta-Llama-3.1-8B-Instruct" or 
                       "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
        """
        # Add validation for model_name
        valid_model_names = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
        ]
        if model_name not in valid_model_names:
            raise ValueError(
                f"Invalid model_name: {model_name}. Must be one of: {valid_model_names}"
            )
            
        self.model_name = model_name
        # Only initialize pipeline if it hasn't been created yet
        global _GLOBAL_PIPELINE
        if _GLOBAL_PIPELINE is None:     
            log.info("Initializing Llama318B pipeline...\n\n")
            
            # # Set PyTorch memory allocation configuration
            # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            # Clear CUDA cache and force garbage collection before loading
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get the number of available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise RuntimeError("No CUDA devices available")
            
            log.info(f"Found {num_gpus} CUDA devices")
            

            _GLOBAL_PIPELINE = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "low_cpu_mem_usage": True,
                },
                device_map="auto",
            )
        self.pipeline = _GLOBAL_PIPELINE

    def _trim_input(
        self, 
        input_text: str | None = None, 
        message_history: List[Dict[str, str]] | None = None,
        max_chars: int = 500000
    ) -> tuple[str | None, List[Dict[str, str]] | None]:
        """Trims either the first user message in history or the input text if no history.
        Uses 500k chars as safe limit for 131k tokens.
        
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
            message_history
        )

        messages = get_single_user_prompt(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            message_history=message_history,
        )
        try:
            # Set do_sample based on temperature
            do_sample = temperature != 0.0
            generation_kwargs = {
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                **kwargs
            }
            
            outputs = self.pipeline(
                messages,
                max_new_tokens=MAX_NEW_TOKENS,
                **generation_kwargs,
            )
            return outputs[0]["generated_text"][-1]["content"]

        except Exception as e:
            raise Exception(f"Exception in calling Llama318B: {e}") 