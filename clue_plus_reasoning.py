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

import pandas as pd
df = pd.read_csv ('dataset/dataset_plus_filtered_liwc.csv')
# simple EDA
# print(df)
# print(df.columns)
print(f'shape: {df.shape}')  # should be 1640 x 6
# print(df.head)

count = 0
for index, row in df.iterrows():
    if row['has_clues']==1:
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
    For now conbine question1 and question 2
    """
    return 'INPUT: ' + row['q1'] + '\n' + row['q2']
    # return 'INPUT: ' + row['q2']

def create_clues(row):
    """
    use the style from the paper
    """
    return 'CLUES'  + ': ' + row['clues']

def create_reasoning(row):
    return 'REASONING' + ': ' + row['reasoning']

def setup_classification(row):
    return('CLASSIFICATION: ' + ('truthful' if row['outcome_class'] == 't' else 'deceptive'))

def create_one_of_k_shots(row):
    k_shot = ''
    print(create_input(row))
    print(create_clues(row))
    print(create_reasoning(row))
    print(setup_classification(row))
    k_shot += create_input(row) + newline
    k_shot += create_clues(row) + newline
    k_shot += create_reasoning(row) + newline
    k_shot += setup_classification(row) + newline
    return k_shot


def setup_inference(row):
    print(create_input(row))
    print('CLUES:')
    print('REASONING: ')
    print('CLASSIFICATION:')
    inference = ''
    inference += create_input(row) + newline
    inference += 'CLUES:' + newline
    inference += 'REASONING: ' + newline
    inference += 'CLASSIFICATION:' + newline
    return inference


def setup_context(inference_row):
    context = ''
    print(create_prelude())
    context += create_prelude()
    for index, row in df.iterrows():
        if row['has_clues']==1:
            # print(row['clues'], row['reasoning'])
            create_one_of_k_shots(row)
            context += create_one_of_k_shots(row)
            print('\n')
            context += newline
    
    context += setup_inference(df.loc[inference_row].copy())
    return context

if __name__ == "__main__":
    final_context = setup_context(inference_row=788)
    print('***********buffer version**********')
    print(final_context)