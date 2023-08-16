"""
LLM generated CLUES and REASONING

Using dataset/sign_events_data_statements.csv with columns: 
signevent,q1,q2,unid,id,outcome_class
Dataset size: [1640 rows x 6 columns]

Adding the CLUES and REASONING column genersted by the LLM and
calling it,
"""


from openai_interface import init_openai, get_chat_completion_with_backoff
from parse_output import parsed_response_str

init_openai()

dataset_used = 'dataset/sign_events_data_statements.csv'

import pandas as pd
df = pd.read_csv (dataset_used)
# simple EDA
print(df.columns)
print(f'shape: {df.shape}')  # should be 1640 x 6
print(df.head)

newline = '\n'
# reasoning_word_limit = 50
# print the prelude
def create_prelude(gt):
    prelude = f"""
In the PARAGRAPH section below: 
First, highlight in the JSON format, words or phrases related to the TRUTHFUL and DECEPTIVE categories.
Treat each sub-category within the two categories in order of decreasing importance.
List each category even if it it's empty.
Next, provide a detailed reason that the paragraph is {gt} based on what you find in the sub-categories below.
Finally, in one word, make a decisive classification on whether the paragraph is TRUTHFUL or DECEPTIVE. 
TRUTHFUL
ingestion - examples are: " dish", "eat", "pizza"
biological-processes - examples are: "eat", "blood", "pain"
numbers - examples are: "second", "thousand", "5", "10"
leisure - examples are: "cook", "chat", "movie" 
future-focus - examples are:  "may", "will", "soon"
DECEPTIVE
apostrophes - examples are, "haven't", "won't", "she's", "can't"
past-tense-focused -  examples are, "ago", "did", "talked", "promised", "gotten"
reward  - examples are, "congratulate", "accomplishment" , "take", "prize", "benefit"
pronouns - examples are, "I", "them", "itself"
personal-pronouns - examples are, "I", "them", "her"
exclamation-mark - example is, "!"
    """

    return prelude + newline

def create_input(row):
    """
    For now, conbine question 1 and question 2
    """
    return 'PARAGRAPH: ' + row['q1'] + '\n' + row['q2']


def construct_context(row, gt):
    """ 
    put the prelude and the response to q1 and q2 
    """
    context = ''
    # print(create_prelude())
    context += create_prelude(gt)
    context += create_input(df.loc[row].copy())
    return context


if __name__ == "__main__":
    ground_truths = []
    predictions = []
    start_row, end_row = 911, 912
    model = 'gpt-4'  # "gpt-3.5-turbo" or "gpt-4"
    print(f'start row: {start_row} end row: {end_row}')
    print(f'Model:{model}')
    for row in range(start_row, end_row):
        print(f"{20*'-'}{row}{20*'-'}")
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        final_context = construct_context(row=row, gt=ground_truth)
        print(final_context)
        print(f'ground truth (GT): {ground_truth}')
        # print(f'INPUT:\n Q1:\n {df.loc[row]["q1"]} \n Q2:\n {df.loc[row]["q2"]}')

        response = get_chat_completion_with_backoff(final_context, model=model)
        print(response + newline)


