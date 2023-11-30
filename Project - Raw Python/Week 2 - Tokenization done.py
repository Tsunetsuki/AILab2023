# CNN
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

# others
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import time

# dataset
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import Flowers102

# read file 
import pandas as pd

# label
from scipy.io import loadmat
import json
from tqdm import tqdm
from itertools import islice

root = './data/'
features_path = os.path.join(root, 'features.csv')
patient_notes_path = os.path.join(root, 'patient_notes.csv')
sample_submission_path = os.path.join(root, 'sample_submission.csv')
test_path = os.path.join(root, 'test.csv')
train_path = os.path.join(root, 'train.csv')
features = pd.read_csv(features_path, sep=',', header=0)
patient_notes = pd.read_csv(patient_notes_path, sep=',', header=0)
train_raw = pd.read_csv(train_path, sep=',', header=0)

import re
def df_string2list_of_ints(df_string: str):
    df_string = df_string.strip("[]")
    if df_string == "":
        return []
    entries = re.split(",|;", df_string)
    entries = [entry.strip(" '") for entry in entries]
    ranges = [tuple(int(num_as_str) for num_as_str in entry.split(" ")) for entry in entries]
    return ranges

# merge data from different spreadsheets
data_merged = train_raw.merge(features, on=['feature_num', 'case_num'], how='left')
data_merged = data_merged.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
data_merged["location"] = data_merged["location"].apply(df_string2list_of_ints)
print(data_merged.head())

# take only columns that could be used for training
train = data_merged[["feature_num", "pn_history", "location", ]]
print(train.head())

# filter training data with no location
train = train[train["location"].apply(lambda row: len(row) != 0)]

print(f'Size of dataset= {len(train)}')


import spacy 
from collections import Counter

# use spacy to tokenize the sentence with english model 
nlp = spacy.load("en_core_web_sm")

# Create vocabulary by getting the most common words across (unique) patient histories
import pickle
import os
from os.path import join as pathjoin

# load or create vocabulary
cache_dir = "cache"
cache_file = pathjoin(cache_dir, "vocab.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        vocab = pickle.load(f)
    print("Vocabulary loaded from cache.")
else:
    print("Found no cached vocabulary. Creating...")
    text_to_count_tokens = ' '.join(train["pn_history"].drop_duplicates())
    # use spacy to tokenize the sentence 
    doc = nlp(text_to_count_tokens)
    # Get the most frequent words, filtering out stop words and punctuation.
    word_freq = Counter(token.text.lower() for token in doc if \
                        not token.is_punct and \
                            not token.is_stop and \
                                not token.is_space)

    most_common_words = word_freq.most_common(5000)
    vocab = {word[0]: idx for idx, word in enumerate(most_common_words)}
    with open(cache_file, "wb") as f:
        pickle.dump(vocab, f)

print("Top 10 words: ", ", ".join(list(vocab)[:10]))


from typing import Dict, List

placeholder_index = 5000


# load or create tokenized patient histories
cache_file = pathjoin(cache_dir, "tokenized_pn_histories.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        tokenized_pn_histories = pickle.load(f)
    print("Tokenized patient histories loaded from cache.")
else:
    print("Found no cached tokenized patient histories. Tokenizing...")
    tokenized_pn_histories: Dict[str, List[str]] = {}
    for pn_history in tqdm(train["pn_history"]):
        indexed_words = []
        if pn_history in tokenized_pn_histories:
            continue
        for token in nlp(pn_history):
            if not token.is_punct and not token.is_stop and not token.is_space:
                word = token.text.lower()
                start_idx = token.idx
                end_idx = token.idx + len(token.text)

                word_as_number = vocab[word] if word in vocab else placeholder_index
                
                indexed_words.append({
                    "word_idx": word_as_number,
                    "start": start_idx,
                    "end": end_idx
                })
                    
        tokenized_pn_histories[pn_history] = indexed_words
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_pn_histories, f)


# reformat data from character ranges to [feature, token, does_token_describe_feature] format

train_tokens_with_scores = dict()
for i, (feature_num, pn_history, location) in tqdm(train.iterrows()):
    tokenized_history = tokenized_pn_histories[pn_history]
    tokens_with_scores = []
    for token in tokenized_history:
        for feature_relevant_range in location:
            token_start, token_end = token["start"], token["end"]
            range_start, range_end = feature_relevant_range[0], feature_relevant_range[1]
            
            percentage_of_token_in_range = max(min(token_end, range_end)+1 - max(token_start, range_start), 0) / (token_end+1 - token_start)
            # if percentage_of_token_in_range > 0:
            #     print(percentage_of_token_in_range, token, feature_relevant_range)
            tokens_with_scores.append({"feature_num": feature_num, "word": token["word_idx"], "score": int(percentage_of_token_in_range > 0.9)})
    train_tokens_with_scores[i] = tokens_with_scores

print("data format:")
train_tokens_with_scores[0][:5]

# TODO: Bring feature label into training data for LSTM!
# - must encode the feature label (one out of 143 numbers) into the LSTM input data to train. How to do it?