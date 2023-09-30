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
from prompt_preludes import load_prompt_prelude
import pprint
from config import Configuration
import pandas as pd

init_openai()
newline = '\n'

config = Configuration()
config.print_config()

df = pd.read_csv (config.dataset_used)
# simple EDA
split_index = 782
dataset_count = df.shape[0]
print(f'shape: {df.shape}')  # should be 1640 x 6
# print(df.head)

# not puting any limit on the 'reasoning' word-count 

def create_prelude(gt):
    prelude = load_prompt_prelude()
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
    gt_truthful_samples = sorted(random.sample(range(0, split_index), config.sample_size))
    gt_deceptive_samples = sorted(random.sample(range(split_index, dataset_count), config.sample_size))
    rows = gt_truthful_samples + gt_deceptive_samples

    print(f'Rows used to generate the clues + reasoning: {rows}')
    out_df = prep_output_df(config.dataset_used, ['contains_clues', 'clues', 'reasoning'])

    print(f'Model:{config.model}')
    ground_truths = []
    predictions = []
    rows_used = []
    errors = []
    
    for row in rows:
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        print(f"{20*'-'} ROW: {row} GT: {ground_truth} {20*'-'}")
        final_context = construct_context(row=row, gt=ground_truth)
        print(f'FINAL CONTEXT:\n{final_context}')

        response = get_chat_completion_with_backoff(final_context, 
                                                    config.system_role, config.model, config.temperature)
        for attempt in range(config.nb_attempts):
            try:
                parsed_data = json.loads(response)
                print(f"{20*'-'}Parsed Response{20*'-'}\n")
                print(f"TRUTHFUL CLUES:\n{pprint.pformat(parsed_data['TRUTHFUL'], width=80)}")
                print(f"DECEPTIVE CLUES:\n{pprint.pformat(parsed_data['DECEPTIVE'], width=80)}") 
                print(f"CLASSIFICATION:\nPREDICTED: {parsed_data['CLASSIFICATION']} GT: {ground_truth}") 
                print(f"REASONING:\n{pprint.pformat(parsed_data['REASONING'], width=80)}")

                out_df = parse_n_write_response(response, out_df, row)
                ground_truths.append(ground_truth)
                predictions.append(str.lower(parsed_data['CLASSIFICATION']))
                rows_used.append(row)
                break
            except:
                if attempt == config.nb_attempts -1:
                    print(f'Failed to parse response in row, NO MORE ATTEMPTS: {row} \nResponse:\n{response}')
                    errors.append(f'Failed to parse response in row, NO MORE ATTEMPTS: {row} \nResponse:\n{response}')
                else:
                    print(f'Failed to parse response in row, RETRYING...: {row} \nResponse:\n{response}')


for ground_truth, prediction, row_used in zip(ground_truths, predictions, rows_used):
    print(f'Row used: {row_used}, GT: {ground_truth}, Pred: {prediction}')

if len(errors) > 0:
    for error in errors:
        print(error)

from sklearn.metrics import f1_score
print('Weighted F1-score:', f1_score(ground_truths, predictions, average='weighted'))
save_output_df(config.classification_file, out_df)
