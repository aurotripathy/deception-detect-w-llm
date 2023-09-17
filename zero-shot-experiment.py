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

init_openai()
newline = '\n'
temperature = 0.5
dataset_used = 'dataset/sign_events_data_statements.csv'
classification_file = 'results/zero-shot-classification-with-clues-reasoning.csv'

import pandas as pd
df = pd.read_csv (dataset_used)
# simple EDA
split_index = 782
dataset_count = df.shape[0]
# print(df.columns)
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
    gt_truthful_samples = sorted(random.sample(range(0, split_index), 10))
    gt_deceptive_samples = sorted(random.sample(range(split_index, dataset_count), 10))
    rows = gt_truthful_samples + gt_deceptive_samples
    print(f'Rows used to generate the clues + reasoning: {rows}')
    out_df = prep_output_df(dataset_used, ['contains_clues', 'clues', 'reasoning'])

    model = 'gpt-3.5-turbo'  # "gpt-3.5-turbo" or "gpt-4"
    print(f'Model:{model}')
    ground_truths = []
    predictions = []
    
    for row in rows:
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        print(f"{20*'-'} ROW:{row} GT: {ground_truth} {20*'-'}")
        final_context = construct_context(row=row, gt=ground_truth)
        print(final_context)
        print(f'ground truth (GT): {ground_truth}')
        # print(f'INPUT:\n Q1:\n {df.loc[row]["q1"]} \n Q2:\n {df.loc[row]["q2"]}')

        response = get_chat_completion_with_backoff(final_context, model, temperature)
        try:
            parsed_data = json.loads(response)
            print(f"{20*'-'}Parsed Response{20*'-'}\n")
            print(parsed_data['TRUTHFUL'])  # 
            print(parsed_data['DECEPTIVE'])  # 
            print(parsed_data['CLASSIFICATION'])  # 
            print(parsed_data['REASONING'])  # 

            out_df = parse_n_write_response(response, out_df, row)
            ground_truths.append(ground_truth)
            predictions.append(parsed_data['CLASSIFICATION'])
        except:
            print(f'Parse failure of response in row: {row} \nFinal context: {final_context} \nResponse: {response}')

for ground_truth, prediction in zip(ground_truths, predictions):
    print(f'GT: {ground_truth}, Pred: {prediction}')

from sklearn.metrics import f1_score
print('Weighted F1-score:', f1_score(ground_truths, predictions, average='weighted'))
save_output_df(classification_file, out_df)
