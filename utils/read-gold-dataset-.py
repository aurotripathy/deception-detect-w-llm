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

if __name__ == "__main__":
    root_dir = '/home/auro/deception/spam-dataset/op_spam_v1.4'
    file_list = collect_all_files(root_dir, 'deceptive')
    for file in file_list:
        print(file)
    print(f'Total files: {len(file_list)}')

    print('dirs')
    dirs =  [path for path in file_list if os.path.isdir(path)]
    print(dirs)

    file_list = collect_all_files(root_dir, 'truthful')
    for file in file_list:
        print(file)
    print(f'Total files: {len(file_list)}')

    print('dirs')
    dirs =  [path for path in file_list if os.path.isdir(path)]
    print(dirs)