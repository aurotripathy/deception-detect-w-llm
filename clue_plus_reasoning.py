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

for index, row in df.iterrows():
    if row['has_clues']==1:
        print(row['clues'], row['reasoning'])

word_limit = 50
# print the prelude
prelude = f"""
This is an overall classifier for truthful and deceptive statements.
First, present CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning,
tones, references) that support the classification determination of the input.
Second, deduce a diagnostic REASONING process from premises (i.e., clues, input) that supports the classification
determination (Limit the number of words to {word_limit}).
Third, determine the overall CLASSIFICATION of INPUT as Truthful or Deceptive considering CLUES, the REASONING
process and the INPUT.
"""
def create_prelude():
    return prelude

def create_input(row):
    pass

def create_clues(row):
    pass

def create_reasoning(row):
    pass

def create_one_of_k_shots(row):
    create_input(row)
    create_clues(row)
    create_reasoning(row)

def setup_inference(row):
    pass

create_prelude()

for index, row in df.iterrows():
    if row['has_clues']==1:
        # print(row['clues'], row['reasoning'])
        create_one_of_k_shots(row)

inference_row = 7  # some random row
setup_inference(inference_row)