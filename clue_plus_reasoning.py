"""
k-shot learning on how to classify thuthfuil adn deceptive statements 
based on hand-written (now llm generated) clues and reasoning
"""

"""
Inspiraton from: 
Text Classification via Large Language Models
Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, Guoyin Wang

In this paper, we introduce Clue And Reasoning Prompting (CARP). 
CARP adopts a progressive reasoning strategy tailored to addressing 
the complex linguistic phenomena involved in text classification: 
CARP first prompts LLMs to find superficial clues 
(e.g., keywords, tones, semantic relations, references, etc), 
based on which a diagnostic reasoning process is induced for final decisions. 

input: <demo-text-1>
clues: <demo-clues-1>
reasoning: <demo-reason-1>
sentiment: <demo-label-word-1>

input: <demo-text-2>
clues: <demo-clues-2>
reasoning: <demo-reason-2>
sentiment: <demo-label-word-2>
... ...
input: <demo-text-n>
clues: <demo-clues-n>
reasoning: <demo-reason-n>
sentiment: <demo-label-word-n>

input: <text>
clues: 
reasoning: 
sentiment: 
"""

from openai_interface import init_openai, get_chat_completion_with_backoff
from parse_output import extract_classification
import json

init_openai()

import pandas as pd
# df = pd.read_csv ('dataset/dataset_plus_filtered_liwc.csv')
df = pd.read_csv ('dataset/llm_generated_clues_reasoning_events_data_statements.csv')

# simple EDA
# print(df)
# print(df.columns)
print(f'shape: {df.shape}')  # should be 1640 x 6
# print(df.head)

count = 0
for index, row in df.iterrows():
    if row['contains_clues'] == 1:
        count += 1
        print(f"[{row['id'] + 1}], {row['clues']}, {row['reasoning']}")
print(f'Count: {count}')

newline = '\n'
reasoning_word_limit = 50
# print the prelude
prelude = f"""
This is an overall classifier for truthful and deceptive statements.
First, present CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning,
tones, references) that support the classification determination of the input.
Second, deduce a diagnostic REASONING process from premises (i.e., clues, input) that supports the classification
determination (Limit the number of words to {reasoning_word_limit}).
Third, determine the overall CLASSIFICATION of INPUT as Truthful or Deceptive considering CLUES, the REASONING
process and the INPUT.
"""
def create_prelude():
    return prelude + newline

def create_input(row):
    """
    For now, combine question 1 and question 2 into a single input
    """
    return 'INPUT: ' + row['q1'] + '\n' + row['q2']

def create_clues(row):
    """
    use the style from the paper
    """
    # return json.dumps({'CLUES' : json.dumps(row['clues'])})
    return json.dumps(json.loads(row['clues']))

def create_reasoning(row):
    return json.dumps(json.loads(row['reasoning']))

def create_classification(row):
    # return dict({'CLASSIFICATION': 'truthful' if row['outcome_class'] == 't' else 'deceptive'})
    return json.dumps(json.loads(row['classification']))

def create_one_of_k_shots(row):
    k_shot = ''
    k_shot += create_input(row) + newline
    k_shot += str(create_clues(row)) + newline
    k_shot += str(create_reasoning(row)) + newline
    k_shot += str(create_classification(row)) + newline
    return k_shot


def setup_inference(row):
    """ we leave CLUEs, REASONING, and CLASSIFICATION blank for the LLM to generate"""
    inference = ''
    inference += create_input(row) + newline
    inference += str(json.dumps(json.loads('{"CLUES" : ""}'))) + newline
    inference += str(json.dumps(json.loads('{"REASONING": ""}'))) + newline
    inference += str(json.dumps(json.loads('{"classification": ""}'))) + newline
    return inference


def construct_context(inference_row, k_shot_count):
    """ constructs the k-shots and the section that has to inferred
        The k is determined by the has_clues flag, this will change...
    """
    context = ''
    # print(create_prelude())
    context += create_prelude()
    count = 0
    for index, row in df.iterrows():
        if row['contains_clues'] == 1:
            # print(row['clues'], row['reasoning'])
            # create_one_of_k_shots(row)
            context += create_one_of_k_shots(row)
            count += 1
            # print('\n')
            context += newline
        if count == k_shot_count:
            break

    context += setup_inference(df.loc[inference_row].copy())
    return context


if __name__ == "__main__":
    k_shot_count = 3
    ground_truths = []
    predictions = []
    start_row, end_row = 900, 920
    model = 'gpt-4'  # "gpt-3.5-turbo" or "gpt-4"
    print(f'start row: {start_row} end row: {end_row}')
    print(f'Model:{model}')
    for row in range(start_row, end_row):
        print(f"{20*'-'}{row} context GT {20*'-'}")
        final_context = construct_context(row, k_shot_count)
        print(final_context)
        ground_truth = 'truthful' if df.loc[row]['outcome_class'] == 't' else 'deceptive'
        print(f'ground truth (GT): {ground_truth}')

        response = get_chat_completion_with_backoff(final_context, model=model)
        print(f"{20*'>'}{row} response GT {20*'>'}")
        print(response + newline)
        
        predicted_class = extract_classification(response)
        ground_truths.append(ground_truth)
        predictions.append(predicted_class)

for ground_truth, prediction in zip(ground_truths, predictions):
    print(f'GT: {ground_truth}, Pred: {prediction}')

from sklearn.metrics import f1_score
print('Weighted F1-score:', f1_score(ground_truths, predictions, average='weighted'))