""" 
llm_genenarted output
adds clues and ressoning
"""
import pandas as pd
 
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

if __name__ == "__main__":
    dataset_used = 'dataset/sign_events_data_statements.csv'
    llm_generated_dataset = 'dataset/llm_generated_clues_reasoning_events_data_statements.csv'  
    df= prep_output_df(source_file=dataset_used, col_names=['contains_clues', 'clues', 'reasoning'])
    save_output_df(llm_generated_dataset, df=df)
