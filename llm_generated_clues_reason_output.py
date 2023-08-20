""" 
llm_genenarted output
adds clues and ressoning
"""
import pandas as pd
import json

def prep_output_df(source_file, col_names):
    df = pd.read_csv(source_file)
    # add three more columns
    for col_name in col_names:
        df[col_name] = ''
    return df

def save_output_df(dest_file, df):
    """ 
    save as csv
    """
    # check the extension is .csv file
    
    if dest_file.endswith('.csv'):
        df.to_csv(dest_file)
    else:
        print(f'Output file {dest_file} must be a csv file')
        exit(1)
    print(f'Saved to {dest_file}')

def parse_n_write_response(response, df, row):
    try:
        parse = json.loads(response)
    except:
        print(f'Encountered a parsing exception!')
    
    print(f'-------------row: {row}----------------')
    print(parse['TRUTHFUL'])
    print(parse['DECEPTIVE'])
    print(parse['REASONING'])
    print(parse['CLASSIFICATION'])
    clues_dict = {"TRUTHFUL": parse['TRUTHFUL'], "DECEPTIVE": parse['DECEPTIVE']}
    print(clues_dict)
    copy_df = df.copy(deep=True)
    copy_df.at[row, 'contains_clues'] = 1
    copy_df.at[row, 'clues'] = clues_dict
    copy_df.at[row, 'reasoning'] = parse['REASONING']
    return copy_df

def create_dummy_response():
    pass
    return """ 
{
  "TRUTHFUL": {
    "ingestion": [],
    "biological-processes": [],
    "numbers": ["two", "one"],
    "leisure": [],
    "future-focus": ["is getting", "will be"]
  },
  "DECEPTIVE": {
    "apostrophes": ["we're"],
    "past-tense-focused": ["lost", "has become", "has taken", "has found"],
    "reward": [],
    "pronouns": ["we", "our", "he", "him", "its", "we", "I", "my", "my", "brothers", "one", "my", "brother", "who"],
    "personal-pronouns": ["we", "our", "he", "him", "its", "we", "I", "my", "my", "brothers", "one", "my", "brother", "who"],
    "exclamation-mark": []
  },
  "REASONING": "The paragraph is truthful because it contains more truthful indicators than deceptive ones. The paragraph contains numbers and future-focus words which are indicators of truthfulness. Although there are some deceptive indicators such as apostrophes, past-tense-focused words, and pronouns, they are not strong enough to classify the paragraph as deceptive.",
  "CLASSIFICATION": "TRUTHFUL"
}
"""


if __name__ == "__main__":
    dataset_used = 'dataset/sign_events_data_statements.csv'
    llm_generated_dataset = 'dataset/llm_generated_clues_reasoning_events_data_statements.csv'  
    df = prep_output_df(source_file=dataset_used, col_names=['contains_clues', 'clues', 'reasoning'])
    response = create_dummy_response()
    df  = parse_n_write_response(response, df, 0)
    save_output_df(llm_generated_dataset, df=df)
