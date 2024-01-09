
from openai import OpenAI
import os
client = OpenAI()

# https://platform.openai.com/docs/guides/rate-limits/error-mitigation
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def init_openai():
    client.api_key = os.environ["OPENAI_API_KEY"]  # source the ~/.zshrc file



def get_chat_completion(prompt, system_role_content, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "system", "content": system_role_content},
                {"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        max_tokens=1000,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    # print(f'>>>Raw Response:\n{response}')
    # print(f'Prompt tokens counted:{response["usage"]["prompt_tokens"]} ')
    # print(f'Completion tokens counted: {response["usage"]["completion_tokens"]}')
    # print(f'Total tokens counted: {response["usage"]["total_tokens"]}')
    print(f'Entire Response"\n{response}')    
    # return response.choices[0].message["content"]
    # return response['choices'][0]['message']['content']
    return response.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat_completion_with_backoff(prompt, system_role, model, temperature):
    return get_chat_completion(prompt, system_role, model, temperature)
