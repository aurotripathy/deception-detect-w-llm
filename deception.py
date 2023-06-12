import pandas as pd
df = pd.read_csv ('sign_events_data_statements.csv')
# simple EDA
print(df)
print(df.columns)
print(f'shape: {df.shape}')  # should be 1640 x 6


def split_classes(df, category):
   return df[df['outcome_class']== category]

truth_df = split_classes(df, 't')
print(truth_df)
print(f'thuth shape: {truth_df.shape}')  # should be 1640 x 6

deceit_df = split_classes(df, 'd')
print(deceit_df)
print(f'deceit shape: {deceit_df.shape}')  # should be 1640 x 6

# pick random non-repeating rows
def randon_non_repeating(df, quantity):
    import random
    rand_df = pd.DataFrame()
    random_list = random.sample(range(df.shape[0]), quantity)
    print("non-repeating random numbers are:")
    return df.iloc[random_list]

nb_samples = 4
random_truth_df = randon_non_repeating(truth_df, nb_samples)
print(f'random truth list:\n, {random_truth_df}')

random_deceit_df = randon_non_repeating(deceit_df, nb_samples)
print(f'random deceit list:\n, {random_deceit_df}')

random_truth_deceit_df = pd.concat([random_truth_df, random_deceit_df])

"""
Q1: Please describe your activity as specific as possible”
Q2:“Which information can you give us to reassure us that you are telling the truth”.
"""

system_role = \
"""
You are a detective presented with the desciption of an activity and supporting information to reassure you that the description is true.
You have to decide whether the description is truthful or deceitful.
"""


def construct_system_message_object():
    return {"role": "system", "content": system_role}

def construct_user_content(row):
    activity_header = 'Activity:'
    activity_description_header = \
    """
    Description of activity by person in very specific language:
    """
    activity_reassurance_header = \
    """
    Provide information to reassure us that the person is telling the truth about their activity:
    """

    new_line = '\n'
    activity = activity_header + new_line + row['signevent'] + new_line
    q1 = activity_description_header + new_line + row['q1'] + new_line
    q2 = activity_reassurance_header + new_line + row['q2'] + new_line
    return {"role": "user", "content": activity + q1 + q2}

def construct_assistent_content(row):
    new_line = '\n'
    outcome = "Outcome:"
    return {"role": "assistant", "content": outcome + new_line + row['outcome_class']}
    

def construct_few_shot_prompt(df, infer_row):
    messages = []
    messages.append(construct_system_message_object())

    for _, row in df.iterrows():
        messages.append(construct_user_content(row))
        messages.append(construct_assistent_content(row))
    
    messages.append(construct_user_content(infer_row))
    return messages

infer_row = df.iloc[783]   # pick a ramdom row
print(f'Predicting the class outcome for:\n{infer_row}')

prompt = construct_few_shot_prompt(random_truth_deceit_df, infer_row=infer_row)
for message in prompt:
    print(f'message: {message}')

# OpenAI request
import openai
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    model=MODEL,
    messages=prompt,
    temperature=0,
)    
        

print(f'ENTIRE RESPONSE:\n {response}')

print(f"Content response:\n {response['choices'][0]['message']['content']}")
