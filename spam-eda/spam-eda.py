"""
spam EDA
understand the basic stats of the words in the dataset
"""

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')

# print("Max Sequence Length:", model.max_seq_length)

import os
import sys

# Add the parent directory of the project to the sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from utils.read_gold_dataset import collect_all_files, read_a_file

root_dir = 'spam-dataset/op_spam_v1.4'
file_path_list = collect_all_files(root_dir, 'deceptive')
total_samples = len(file_path_list)
print(f'Total samples:{total_samples}')

# import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# doc = nlp('''\
# I wondered if I could write a program to automatically catch clumsy style mistakes I often make.
# I'll try using spaCy!
# It turns out style-checking is a little complicated, so this post is actually just about spaCy.
# ''')

# doc = nlp('hello, hello, hello!')
# print('''
# words\t\t{num_words}
# sentences\t{num_sent}
# '''.format(
#     num_words=len(doc),  # grab number of tokens
#     num_sent=len(list(doc.sents)),
# ))

import numpy as np

total_words = 0
total_sentences = 0
max_words_per_post = -np.inf
min_words_per_post = np.inf
count_greater_that_192_words = 0

for i, file_path in enumerate(file_path_list):
    gt_sentiment, gt_veracity, text = read_a_file(file_path)
    doc = nlp(text)
    word_count = len(doc)
    if word_count > max_words_per_post:
        max_words_per_post = word_count
    if word_count < min_words_per_post:
        min_words_per_post = word_count
    if word_count > 192:
        count_greater_that_192_words += 1
    total_words += word_count
    total_sentences += len(list(doc.sents))
print(f'Average words per post: {total_words/len(file_path_list)}')
print(f'Average sentences per post: {total_sentences/len(file_path_list)}')
print(f'Min/Max words per post: {min_words_per_post}/{max_words_per_post}')
print(f'Sentences with word count greater than 200: {count_greater_that_192_words}')


