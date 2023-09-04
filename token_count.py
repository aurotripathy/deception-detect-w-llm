import tiktoken

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def nb_tokens_in_prompt(prompt, model="gpt-3.5-turbo-0613") -> int:
    """Return the number of tokens is a prompt
    Per docs, "Consider the counts from the function below an estimate, not a timeless guarantee."
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-0301",    
        }:
        num_tokens = len(encoding.encode(prompt))
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        num_tokens = len(encoding.encode(prompt))
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        num_tokens = len(encoding.encode(prompt))
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    return num_tokens


# let's verify the function above matches the OpenAI API response
if __name__ == "__main__":

    import openai

    example_messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
        },
    ]

    prompt = """
I have three tasks:
Medium Priority: feed the dog
Low Priority: Feed the cat
High Priority: feed the lion
What shall I do first and next
"""
    for model in [
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4",
        ]:
        print(model)
        # Example Token Count From The Function Defined Above
        # print(F"{num_tokens_from_messages(example_messages, model)} Prompt Tokens Counted By Num_Tokens_From_Messages().")
        print(F"{nb_tokens_in_prompt(prompt, model)} Prompt Tokens Counted by tiktoken.")
        # Example Token Count From The Openai Api
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=1
        )
        # response = openai.ChatCompletion.create(
        #     model = model,
        #     messages = example_messages,
        #     temperature=0,
        #     max_tokens=1,  # We're Only Counting Input Tokens Here, So Let's Not Waste Tokens On The Output
        # )
        print(f'Response:\n{response}')
        print(f'{response["usage"]["prompt_tokens"]} Prompt Tokens Counted By The Openai Api.')
        print(f'{response["usage"]["completion_tokens"]} Completion Tokens Counted By The Openai Api.')
        print(f'{response["usage"]["total_tokens"]} Total Tokens Counted By The Openai Api.')
        print()    