"""
creatng a dict out of the response
very brittle for now, need to fix
"""

import re
import json

# def parsed_response_str(response):
#     delims = ['CLUES:', '\nREASONING:', '\nCLASSIFICATION:']
#     delims_clean = ['CLUES', 'REASONING', 'CLASSIFICATION']
#     pattern = '|'.join(delims)
#     result = re.split(pattern, response)
#     result_list = list(filter(None, result))  # filter out the initial empty string
#     # print(result_list)
#     # print(delims)
#     if len(result_list) != len(delims):
#         print(len(result_list), len(delims))
#         exit('something went wrong')
#     else:
#         split_dict = {}
#         for delim, value in zip(delims_clean, result_list):
#             split_dict[delim] = value
#     return split_dict 

def extract_classification(response):
    response_array = response.split('\n')
    return json.loads(response_array[2])['classification']


if __name__ == "__main__":
    response = """
CLUES: Future focused: "coming up today"
REASONING: The response contains a future focused statement "coming up today" and is assertive, both are indicators of truthfulness
CLASSIFICATION: truthful"""

    print(type(response))

    print(parsed_response_str(response))

