"""
The premis: Research proves LIWC to be successful in detecting deception. 
"Deceptive statements compared with truthful ones are:
- moderately descriptive, 
- distanced from self, and 
- more negative" (Tausczik & Pennebaker, 2010). 
Such a description is not surprising, considering that more information carries a greater risk of
uncovering the truth. By analyzing deception in synchronous computer-mediated communication, it was shown how the linguistic style of the sender (who
lies) and the receiver (who is unaware of the deception) changes. Both respondents were using more words overall (especially sensory) and fewer 1st person
pronouns during deception compared to honest interaction (Hancock et al., 2005; Zhou et al., 2004). Thus, it is obvious that linguistic style hides patterns
that are specific to true and false communication, which can to some extent be successfully detected using the LIWC approach. Since LIWC software lacks
context analysis, it is recommended to combine it with other natural language processing methods. Based on previous research, LIWC, together with an analysis
combined with n-grams achieved satisfactory results (Feng et al., 2012; Ott et al., 2011). Given the large number of dimensions that LIWC possesses, a
selection of important features needs to be done, in order to prevent overfitting and maximize the performance of the machine learnomg model
from "Detecting Deception using Natural Language Processing and Machine
Learning in Datasets on Covid-19 and Climate Change"
"""

import random
import json
from utils.openai_interface import init_openai, get_chat_completion_with_backoff
from config import Configuration

context_prelude = """
Any one of the criteria below is sufficient to label a statement as 'deceptive':
1. Deceptive statements are low to moderately descriptive.
2. Deceptive statements distance themselves from self. 
3. Deceptive statements are moderately negative.
4. Deceptive statements are overwhelmingly positive with exclamation amrks and capital letters

Is the STATEMENT below deceptive or truthful?
Reason step by step.
Apply each criteria to the STATEMENT below. 
Count the number of times the criteria has been met.

Your response must be in the JSON format with three keys, 'analysis', 'criteria-count', and 'sentiment'.
The 'analysis' key has value containing your detailed analysis of each criteria and whether the criteria has been met. 
The criteria keys are 'moderately-descriptive', 'distancing-from-self', 'moderately-negative', 'overwhelmingly-positive'.
Each criteria key has two sub-keys, a 'rationale' containing the reason and a sub-key named 'criteria-met' with value either 'true' or 'false'
The 'count' key must contain a numeric value that is the count of the times the criteria has been met.  
The 'veracity' key has value containing either 'truthful' or 'deceptive' strictly based on the criteria set above. 
If any criteria is met, you must declare the statement is 'deceptive'. 
"""

import random
import json
from utils.openai_interface import init_openai, get_chat_completion_with_backoff
from utils.llm_generated_clues_reason_output import prep_output_df, save_output_df, parse_n_write_response
import pprint
from config import Configuration
from utils.read_gold_dataset import collect_all_files, read_a_file

init_openai()
newline = '\n'

config = Configuration(temperature=1, sample_size=5)
config.print_config()

def create_prelude():
    prelude = context_prelude
    return prelude + newline

def create_input(file_path):
    """
    For now, combine question 1 and question 2
    Thre may be a need to weight them unequally
    """
    _, _, text = read_a_file(file_path)
    return 'STATEMENT:\n ' + text

def construct_context(file_path):
    """ 
    put the prelude and the response to q1 and q2 
    """
    context = ''
    context += create_prelude()
    context += create_input(file_path)
    return context


if __name__ == "__main__":
    root_dir = '/home/auro/deception/spam-dataset/op_spam_v1.4'
    file_path_list = collect_all_files(root_dir, 'truthful')
    total_samples = len(file_path_list)

    sample_indices = sorted(random.sample(range(0, total_samples), config.sample_size))
    
    print(f'Files indices used in the analysis: {sample_indices}')

    ground_truths = []
    predictions = []
    files_used = []
    errors = []
    
    for sample in sample_indices:
        gt_sentiment, gt_veracity, text = read_a_file(file_path_list[sample])
        
        final_context = construct_context(file_path_list[sample])

        print(f'FINAL CONTEXT:\n{final_context}')

        for attempt in range(config.nb_attempts):
            response = get_chat_completion_with_backoff(final_context, 
                                                        config.system_role, config.model, config.temperature)
            print(response)
            print(f'{10*"-"}')
            print(f'Ground Truth:: sentiment: {gt_sentiment}, veracity: {gt_veracity}')
            try:  # valid json
                parsed_data = json.loads(response)
                break
            except:
                if attempt == config.nb_attempts -1:
                    print(f'Failed to parse response in row, NO MORE ATTEMPTS: {sample} \nResponse:\n{response}')
                    errors.append(f'Failed to parse response in row, NO MORE ATTEMPTS: {sample} \nResponse:\n{response}')
                else:
                    print(f'Failed to parse response in row, RETRYING...: {sample} \nResponse:\n{response}')


for ground_truth, prediction, row_used in zip(ground_truths, predictions, files_used):
    print(f'Row used: {row_used}, GT: {ground_truth}, Pred: {prediction}')

if len(errors) > 0:
    for error in errors:
        print(error)

