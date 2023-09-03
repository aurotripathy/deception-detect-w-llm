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

# let's verify the function above matches the OpenAI API response
if __name__ == "__main__":

    import openai

    example_messages = [
        {
            "Role": "System",
            "Content": "You Are A Helpful, Pattern-Following Assistant That Translates Corporate Jargon Into Plain English.",
        },
        {
            "Role": "System",
            "Name": "Example_User",
            "Content": "New Synergies Will Help Drive Top-Line Growth.",
        },
        {
            "Role": "System",
            "Name": "Example_Assistant",
            "Content": "Things Working Well Together Will Increase Revenue.",
        },
        {
            "Role": "System",
            "Name": "Example_User",
            "Content": "Let's Circle Back When We Have More Bandwidth To Touch Base On Opportunities For Increased Leverage.",
        },
        {
            "Role": "System",
            "Name": "Example_Assistant",
            "Content": "Let's Talk Later When We're Less Busy About How To Do Better.",
        },
        {
            "Role": "User",
            "Content": "This Late Pivot Means We Don't Have Time To Boil The Ocean For The Client Deliverable.",
        },
    ]

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
        print(F"{num_tokens_from_messages(example_messages, model)} Prompt Tokens Counted By Num_Tokens_From_Messages().")
        # Example Token Count From The Openai Api
        response = openai.ChatCompletion.create(
            model = model,
            messages = example_messages,
            Temperature=0,
            Max_Tokens=1,  # We're Only Counting Input Tokens Here, So Let's Not Waste Tokens On The Output
        )
        print(F'{response["Usage"]["Prompt_Tokens"]} Prompt Tokens Counted By The Openai Api.')
        print()    