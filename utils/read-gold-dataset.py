"""
Return the gold dataset filelist so you can read content later

(base) auro@auro-All-Series:~/deception/spam-dataset/op_spam_v1.4$ tree -L 2 negative_polarity/ 
negative_polarity/
├── deceptive_from_MTurk
│   ├── fold1
│   ├── fold2
│   ├── fold3
│   ├── fold4
│   └── fold5
└── truthful_from_Web
    ├── fold1
    ├── fold2
    ├── fold3
    ├── fold4
    └── fold5

12 directories, 0 files
(base) auro@auro-All-Series:~/deception/spam-dataset/op_spam_v1.4$ tree -L 2 positive_polarity/
positive_polarity/
├── deceptive_from_MTurk
│   ├── fold1
│   ├── fold2
│   ├── fold3
│   ├── fold4
│   └── fold5
└── truthful_from_TripAdvisor
    ├── fold1
    ├── fold2
    ├── fold3
    ├── fold4
    └── fold5


Note: We are agnostic on the polarity and focus our attention on truthful/deceptive
Papers: https://www.cs.cmu.edu/~hovy/papers/14ACL-deceptive-opinions.pdf
"""


import glob
import os
import re

def collect_all_files_and_dirs(directory):
    """
    Recursively collect all file names in a directory and its subdirectories.
    Includes dir names as well
    """
    return glob.glob(directory + '/**/*', recursive=True)

def collect_all_files(directory, attribute):
    """
    Recursively collect all file names in a directory and its subdirectories.
    Apply 'attribute' filter
    Does not include dir names.
    Pick up only text files
    """
    if attribute in ['truthful', 'deceptive']:
        all_paths = glob.glob(directory + '/**/*.txt', recursive=True)
        return [path for path in all_paths if not os.path.isdir(path) and attribute in path]
    else:
        print("Incorrect attribute! Must match 'truthful' or deceptive'")
        exit(1)

def read_a_file(file_path):
    """
    In addition to returning the text,
    also returns the polarity and the thuthfulness (or deception)
    """
    
    attributes = re.findall(r'positive|negative|truthful|deceptive', file_path, re.IGNORECASE)
    if len(attributes) == 0:
        print(f'Opps! file path does not have proper attributes: {file_path}')
    with open(file_path) as fp:
        text = fp.read()
    return attributes[0], attributes[1], text
    

if __name__ == "__main__":
    root_dir = '/home/auro/deception/spam-dataset/op_spam_v1.4'
    file_list = collect_all_files(root_dir, 'truthful')
    for file in file_list:
        print(file)
    print(f'Total files: {len(file_list)}')

    print('dirs')
    dirs =  [path for path in file_list if os.path.isdir(path)]
    print(dirs)
    
    print(f'file name:{file_list[0]}')
    for file_path in file_list:
        polarity, truthfulness, text = read_a_file(file_path)
        print(f'polarity: {polarity} truthfulness: {truthfulness} file name: {file_path}\n{text}')

    file_list = collect_all_files(root_dir, 'truthful')
    for file in file_list:
        print(file)
    print(f'Total files: {len(file_list)}')

    print('dirs')
    dirs =  [path for path in file_list if os.path.isdir(path)]
    print(dirs)

    print(f'file name:{file_list[0]}')
    for file_path in file_list:
        polarity, truthfulness, text = read_a_file(file_path)
        print(f'polarity: {polarity} truthfulness: {truthfulness} file name: {file_path}\n{text}')
