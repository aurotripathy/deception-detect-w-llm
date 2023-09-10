"""
zero-shot experiment
"""

from openai_interface import init_openai, get_chat_completion_with_backoff
from llm_generated_clues_reason_output import prep_output_df, save_output_df, parse_n_write_response

init_openai()

dataset_used = 'dataset/sign_events_data_statements.csv'
classification_file = 'results/zero-shot-classification-with-clues-reasoning.csv'

import pandas as pd
df = pd.read_csv (dataset_used)
# simple EDA
# print(df.columns)
# print(f'shape: {df.shape}')  # should be 1640 x 6
# print(df.head)

newline = '\n'
# not puting any limit on the 'reasoning' word-count 

def create_prelude(gt):
    prelude = """
In the PARAGRAPH section below:
First, extract words or phrases related to the TRUTHFUL and DECEPTIVE categories and sub-categories using the EXAMPLE section as a guide.
Next, provide detailed reasons as to why the PARAGRAPH is TRUTHFUL or DECEPTIVE based on values you extracted in each sub-categories. 
Pay special attention to 'importance' to decide which class the PARAGRAPH belongs to. 
Treat each sub-category within the two categories based on the 'importance' key.
Finally, in one word, make a final classification on whether the paragraph is TRUTHFUL or DECEPTIVE. 
Generate the response in only the JSON format with keys, "TRUTHFUL", "DECEPTIVE", "REASONING", "CLASSIFICATION".
In the response, list each sub-category even if it's empty and include its importance.
The key for the subcategory should be 'extracted' 
The CLASSIFICATION key can have only two values, 'truthful' or 'deceptive'

EXAMPLE: TRUTHFUL and DECEPTIVE CATEGORIES and SUB-CATEGORIES and IMPORTANCE of each SUB-CATEGORY
{"TRUTHFUL": 
    {"ingestion": { "examples": ["dish", "eat", "pizza"], "importance": "high" }, 
     "biological-processes": { "examples": ["eat", "blood", "pain"], "importance": "high" }, 
     "numbers": { "examples": ["second", "thousand", "5", "10"], "importance": "medium" }, 
     "leisure": { "examples": ["cook", "chat", "movie"], "importance": "medium" }, 
     "future-focus": { "examples": ["may", "will", "soon"], "importance": "medium" } 
    }, 
 "DECEPTIVE": 
    {"apostrophes": { "examples": ["haven't", "won't", "she's", "can't"], "importance": "high" }, 
     "past-tense-focused": { "examples": ["ago", "did", "talked", "promised", "gotten"], "importance": "high" }, 
     "reward": { "examples": ["congratulate", "accomplishment", "take", "prize", "benefit"], "importance": "high" }, 
     "pronouns": { "examples": ["I", "them", "itself"], "importance": "high" }, 
     "personal-pronouns": { "examples": ["I", "them", "her"], "importance": "high" }, 
     "exclamation-mark": { "examples": ["!"], "importance": "medium" } 
    } 
} 
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
    truth_start_row, truth_end_row = 0, 0
    deceptive_start_row, deceptive_end_row = 789, 790
    rows = list(range(truth_start_row, truth_end_row))
    rows += list(range(deceptive_start_row, deceptive_end_row))
    print(f'Rows used to generate the clues + reasoning: {rows}')
    out_df = prep_output_df(dataset_used, ['contains_clues', 'clues', 'reasoning'])

    model = 'gpt-4-0613'  # "gpt-3.5-turbo" or "gpt-4"
    print(f'Model:{model}')
    for row in rows:
        print(f"{20*'-'}{row}{20*'-'}")
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        final_context = construct_context(row=row, gt=ground_truth)
        print(final_context)
        print(f'ground truth (GT): {ground_truth}')
        # print(f'INPUT:\n Q1:\n {df.loc[row]["q1"]} \n Q2:\n {df.loc[row]["q2"]}')

        response = get_chat_completion_with_backoff(final_context, model=model)
        print(f'Response---------------------\n')
        print(response + newline)
        out_df = parse_n_write_response(response, out_df, row)

save_output_df(classification_file, out_df)
