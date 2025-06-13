import time
import openai
import urllib.request
from colorama import Fore, Style
from typing import List, Dict

from ddxdriver.logger import log


def get_chat_messages(
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    message_history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """
    Input: user prompt, system prompt, message history already in chat format
        - Assumes that a given message history already contains a system prompt, will error if you input both system prompt and message_history
    Output: chat messages formatted as chat completion
    """
    assert not (
        message_history and system_prompt
    ), "system_prompt should be None if message_history is provided"

    if message_history:
        messages = message_history
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
    elif system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
    elif user_prompt:
        messages = [
            {"role": "user", "content": user_prompt},
        ]
    else:
        raise ValueError("All parameters are None, at least one should be specified")

    return messages


def retry_with_exponential_backoff(
    func,
    max_retries=3,
    initial_delay=1,
    exponential_base=1.5,
    errors_tuple=(Exception,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors_tuple as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if isinstance(e, ValueError) or (num_retries > max_retries):
                    log.info(
                        Fore.RED
                        + f"ValueError / Maximum number of retries ({max_retries}) exceeded."
                        + Style.RESET_ALL
                    )
                    result = "error:{}".format(e)
                    if (
                        "Azure has not provided the response due to a content filter being triggered"
                        in result
                    ):
                        log.info(
                            Fore.RED
                            + f'Azure content filter triggered for error: {result}.\nReturning message content as "" in lieu of error message.'
                            + Style.RESET_ALL
                        )
                        return ""
                    prompt = kwargs.get("user_prompt", args[0] if len(args) > 0 else None)
                    res_info = {
                        "input": prompt,
                        "output": result,
                        "num_input_tokens": (len(prompt) // 4 if prompt else 0),  # approximation
                        "num_output_tokens": 0,
                        "logprobs": [],
                    }
                    return result, res_info
                # Sleep for the delay
                log.info(
                    Fore.YELLOW
                    + f"Error encountered ({e}). Retry ({num_retries}) after {delay} seconds..."
                    + Style.RESET_ALL
                )
                time.sleep(delay)
                # Increment the delay
                delay *= exponential_base
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def get_condensed_chat_messages(
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    message_history: List[Dict[str, str]] | None = None,
) -> List[Dict[str, str]]:
    """
    Similar to get_chat_messages, but concatenates all message history into a single user prompt.
    Useful for models that don't support full chat history.
    """
    # Combine message history into single string if it exists
    combined_prompt = ""
    if message_history:
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]
            if role != "system":  # Skip system messages in the concatenation
                combined_prompt += f"{role.capitalize()}: {content}\n\n"
    
    # Add final user prompt if provided
    if user_prompt:
        combined_prompt += f"User: {user_prompt}"
    
    # Create messages list with optional system prompt and combined user prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if combined_prompt:
        messages.append({"role": "user", "content": combined_prompt.strip()})
    elif not system_prompt:
        raise ValueError("All parameters are None, at least one should be specified")
    
    return messages


def get_single_user_prompt(
    user_prompt: str | None = None,
    system_prompt: str | None = None,
    message_history: List[Dict[str, str]] | None = None,
) -> str:
    """
    Combines all messages (system prompt, message history, and user prompt) into a single string.
    Format: 
    System: <system_prompt>
    User: <message1>
    Assistant: <message2>
    User: <message3>
    ...
    User: <final_user_prompt>
    
    Returns an empty string if all inputs are None.
    """
    combined_prompt = []
    
    # Add system prompt if provided
    if system_prompt:
        combined_prompt.append(f"System: {system_prompt}")
    
    # Add message history
    if message_history:
        for msg in message_history:
            role = msg["role"]
            content = msg["content"]
            combined_prompt.append(f"{role.capitalize()}: {content}")
    
    # Add final user prompt if provided
    if user_prompt:
        combined_prompt.append(f"User: {user_prompt}")
    
    return [{"role": "user", "content": "\n\n".join(combined_prompt)}]