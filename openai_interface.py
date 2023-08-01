
import openai
import os

# https://platform.openai.com/docs/guides/rate-limits/error-mitigation
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def init_openai():
    openai.api_key = os.environ["OPENAI_API_KEY"]  # source the ~/.zshrc file


def get_chat_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat_completion_with_backoff(prompt, model="gpt-3.5-turbo"):
    return get_chat_completion(prompt, model)
