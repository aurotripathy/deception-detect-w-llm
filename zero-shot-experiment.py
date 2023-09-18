"""
zero-shot experiment

Using dataset/sign_events_data_statements.csv with columns: 
signevent,q1,q2,unid,id,outcome_class
Dataset size: [1640 rows x 6 columns]
TRUTHFUL data [:782]
DECEPTIVE data [782:]

"""
import random
import json
from utils.openai_interface import init_openai, get_chat_completion_with_backoff
from utils.llm_generated_clues_reason_output import prep_output_df, save_output_df, parse_n_write_response
import pprint

init_openai()
newline = '\n'
model = 'gpt-4'  # "gpt-3.5-turbo" or "gpt-4"
temperature = 0.5
nb_attempts = 2
sample_size = 5
dataset_used = 'dataset/sign_events_data_statements.csv'
classification_file = 'results/zero-shot-classification-with-clues-reasoning.csv'

import pandas as pd
df = pd.read_csv (dataset_used)
# simple EDA
split_index = 782
dataset_count = df.shape[0]
print(f'shape: {df.shape}')  # should be 1640 x 6
# print(df.head)

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
The key for the subcategory must be 'extracted' 
The CLASSIFICATION key can have only two values, 'truthful' or 'deceptive'

EXAMPLE: TRUTHFUL and DECEPTIVE CATEGORIES and SUB-CATEGORIES and the IMPORTANCE of each SUB-CATEGORY
{"TRUTHFUL": 
    {"ingestion": { "examples": ["dish", "eat", "pizza"], "importance": "high" }, 
     "biological-processes": { "examples": ["eat", "blood", "pain"], "importance": "medium" }, 
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
     "exclamation-mark": { "examples": ["!"], "importance": "high" } 
    } 
} 
    """

    return prelude + newline

def create_input(row):
    """
    For now, combine question 1 and question 2
    Thre may be a need to weight them unequally
    """
    return 'PARAGRAPH:\n ' + row['q1'] + '\n' + row['q2']

def construct_context(row, gt):
    """ 
    put the prelude and the response to q1 and q2 
    """
    context = ''
    context += create_prelude(gt)
    context += create_input(df.loc[row].copy())
    return context


if __name__ == "__main__":
    ground_truths = []
    gt_truthful_samples = sorted(random.sample(range(0, split_index), sample_size))
    gt_deceptive_samples = sorted(random.sample(range(split_index, dataset_count), sample_size))
    rows = gt_truthful_samples + gt_deceptive_samples
    print(f'Rows used to generate the clues + reasoning: {rows}')
    out_df = prep_output_df(dataset_used, ['contains_clues', 'clues', 'reasoning'])

    print(f'Model:{model}')
    ground_truths = []
    predictions = []
    errors = []
    
    for row in rows:
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        print(f"{20*'-'} ROW: {row} GT: {ground_truth} {20*'-'}")
        final_context = construct_context(row=row, gt=ground_truth)
        print(f'FINAL CONTEXT:\n{final_context}')

        response = get_chat_completion_with_backoff(final_context, model, temperature)
        for attempt in range(nb_attempts):
            try:
                parsed_data = json.loads(response)
                print(f"{20*'-'}Parsed Response{20*'-'}\n")
                print(f"TRUTHFUL CLUES:\n{pprint.pformat(parsed_data['TRUTHFUL'], width=80)}")
                print(f"DECEPTIVE CLUES:\n{pprint.pformat(parsed_data['DECEPTIVE'], width=80)}") 
                print(f"CLASSIFICATION:\nPREDICTED: {parsed_data['CLASSIFICATION']} GT: {ground_truth}") 
                print(f"REASONING:\n{pprint.pformat(parsed_data['REASONING'], width=80)}")

                out_df = parse_n_write_response(response, out_df, row)
                ground_truths.append(ground_truth)
                predictions.append(parsed_data['CLASSIFICATION'])
                break
            except:
                if attempt == nb_attempts -1:
                    print(f'Failed to parse response in row, NO MORE ATTEMPTS: {row} \nResponse:\n{response}')
                    errors.append(f'Failed to parse response in row, NO MORE ATTEMPTS: {row} \nResponse:\n{response}')
                else:
                    print(f'Failed to parse response in row, RETRYING...: {row} \nResponse:\n{response}')


for ground_truth, prediction in zip(ground_truths, predictions):
    print(f'GT: {ground_truth}, Pred: {prediction}')

if len(errors) > 0:
    for error in errors:
        print(error)

from sklearn.metrics import f1_score
print('Weighted F1-score:', f1_score(ground_truths, predictions, average='weighted'))
save_output_df(classification_file, out_df)
