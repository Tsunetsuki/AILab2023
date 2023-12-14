# %%
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

# %% [markdown]
# # Use TweetEval emotion recognition dataset 

# %%
# root = '../../Data/tweeteval/datasets/emotion/'
# mapping_file = os.path.join(root, 'mapping.txt')
# test_labels_file = os.path.join(root, 'test_labels.txt')
# test_text_file = os.path.join(root, 'test_text.txt')
# train_labels_file = os.path.join(root, 'train_labels.txt')
# train_text_file = os.path.join(root, 'train_text.txt')
# val_labels_file = os.path.join(root, 'val_labels.txt')
# val_text_file = os.path.join(root, 'val_text.txt')

# %%
# mapping_pd = pd.read_csv(mapping_file, sep='\t', header=None)
# test_label_pd = pd.read_csv(test_labels_file, sep='\t', header=None)
# test_dataset = open(test_text_file).read().split('\n')[:-1] # remove last empty line 
# train_label_pd = pd.read_csv(train_labels_file, sep='\t', header=None)
# train_dataset = open(train_text_file).read().split('\n')[:-1] # remove last empty line
# val_label_pd = pd.read_csv(val_labels_file, sep='\t', header=None)
# val_dataset = open(val_text_file).read().split('\n')[:-1] # remove last empty line

# %% [markdown]
# # Preprocess training data
# - Given: Notes with ranges and labels
# - Transform into label + lists of tokens with [does token describe label]

# %%
root = './data/'
features_path = os.path.join(root, 'features.csv')
patient_notes_path = os.path.join(root, 'patient_notes.csv')
sample_submission_path = os.path.join(root, 'sample_submission.csv')
test_path = os.path.join(root, 'test.csv')
train_path = os.path.join(root, 'train.csv')
features = pd.read_csv(features_path, sep=',', header=0)
patient_notes = pd.read_csv(patient_notes_path, sep=',', header=0)
train_raw = pd.read_csv(train_path, sep=',', header=0)


# %%
# unusual_numbers = features["feature_num"].value_counts()[features["feature_num"].value_counts() != 1]
# unusual_numbers
features[features["feature_text"] == "Female"]
# features["feature_num"] == 

# %% [markdown]
# ## intro 
# - `case_num`: 0~9, each num belongs their groups ... ? 
# - `pn_num`: the id in patient_notes.csv which is 'pn_history', present the note of each case 
# - `feature_num`: the id in features.csv which is 'feature_num', present the feature of each case 
# - `location`: 

# %%
import re
def df_string2list_of_ints(df_string: str):
    df_string = df_string.strip("[]")
    if df_string == "":
        return []
    entries = re.split(",|;", df_string)
    entries = [entry.strip(" '") for entry in entries]
    ranges = [tuple(int(num_as_str) for num_as_str in entry.split(" ")) for entry in entries]
    return ranges

# %%
train_raw

# %%
data_merged = train_raw.merge(features, on=['feature_num', 'case_num'], how='left')
data_merged = data_merged.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
data_merged["location"] = data_merged["location"].apply(df_string2list_of_ints)
data_merged.head()

# %%
train = data_merged[["feature_text", "pn_history", "location", ]]
train.head()

# %%
# filter training data with no location
train = train[train["location"].apply(lambda row: len(row) != 0)]

# %%
print(f'Size of dataset= {len(train)}')

# %% [markdown]
# ## Tokenization
# - Use spaCy to split the notes into words.
# 
# Before start using spaCy
# ```
# conda install -c conda-forge spacy
# python -m spacy download en_core_web_sm
# ```

# %%
import spacy 
from collections import Counter

# use spacy to tokenize the sentence with english model 
nlp = spacy.load("en_core_web_sm")


# %%
from typing import List, Iterable

def build_vocab_from_lines(lines: Iterable[str]):
    text_to_count_tokens = ' '.join(lines)
    doc = nlp(text_to_count_tokens)
    # Get the most frequent words, filtering out stop words and punctuation.
    word_freq = Counter(token.text.lower() for token in doc if \
                        not token.is_punct and \
                            not token.is_stop and \
                                not token.is_space)
    return word_freq.most_common()

# %%
# Create vocabulary by getting the most common words across (unique) patient histories
import pickle
import os
from os.path import join as pathjoin

cache_dir = "cache"
cache_file = pathjoin(cache_dir, "vocab.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        pn_history_vocab = pickle.load(f)
    print("Vocabulary loaded from cache.")
else:
    print("Found no cached vocabulary. Creating...")
    
    most_common_words = build_vocab_from_lines(train["pn_history"].drop_duplicates())[:5000]
    
    pn_history_vocab = {word[0]: idx for idx, word in enumerate(most_common_words)}
    with open(cache_file, "wb") as f:
        pickle.dump(pn_history_vocab, f)

print("Top 10 words: ", ", ".join(list(pn_history_vocab)[:10]))
# [(k, v) for k, v in vocab.items() if v == 0]

# %%
# Create vocabulary by getting the most common words across (unique) patient histories
import pickle
import os
from os.path import join as pathjoin

cache_dir = "cache"
cache_file = pathjoin(cache_dir, "feature_vocab.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        feature_vocab = pickle.load(f)
    print("Feature vocabulary loaded from cache.")
else:
    print("Found no cached feature vocabulary. Creating...")
    
    most_common_words = build_vocab_from_lines(train["feature_text"].drop_duplicates())[:5000]
    
    feature_vocab = {word[0]: idx for idx, word in enumerate(most_common_words)}
    with open(cache_file, "wb") as f:
        pickle.dump(feature_vocab, f)

print("Top 10 words: ", ", ".join(list(feature_vocab)[:10]))
# [(k, v) for k, v in vocab.items() if v == 0]

# %%
from typing import Dict, List

placeholder_index = 5000

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

                word_as_number = pn_history_vocab[word] if word in pn_history_vocab else placeholder_index
                
                indexed_words.append({
                    "word_idx": word_as_number,
                    "start": start_idx,
                    "end": end_idx
                })
                    
        tokenized_pn_histories[pn_history] = indexed_words
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_pn_histories, f)


# %%
from typing import Dict, List

placeholder_index = len(feature_vocab)

cache_file = pathjoin(cache_dir, "tokenized_features.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        tokenized_features = pickle.load(f)
    print("Tokenized features loaded from cache.")
else:
    print("Found no cached tokenized features. Tokenizing...")
    tokenized_features: Dict[str, List[str]] = {}
    for feature_text in tqdm(train["feature_text"]):
        indexed_words = []
        if feature_text in tokenized_features:
            continue
        for token in nlp(feature_text):
            if not token.is_punct and not token.is_stop and not token.is_space:
                word = token.text.lower()
                word_as_number = feature_vocab[word] if word in feature_vocab else placeholder_index
                
                indexed_words.append(word_as_number)
                    
        tokenized_features[feature_text] = indexed_words
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_features, f)


# %%
tokenized_features

# %% [markdown]
# - Follow the example described here. Use the same architecture, but:
#   - only use the last output of the LSTM in the loss function
#   - use an embedding dim of 128
#   - use a hidden dim of 256.  

# %% [markdown]
# ## Get feature-relevancy of tokens via char ranges

# %%
train_data_preprocessed = dict()
for i, (feature_text, pn_history, location) in tqdm(train.iterrows()):
    tokenized_history = tokenized_pn_histories[pn_history]
    tokens_with_scores = []
    for token in tokenized_history:
        for feature_relevant_range in location:
            token_start, token_end = token["start"], token["end"]
            range_start, range_end = feature_relevant_range[0], feature_relevant_range[1]
            
            percentage_of_token_in_range = max(min(token_end, range_end)+1 - max(token_start, range_start), 0) / (token_end+1 - token_start)
            # if percentage_of_token_in_range > 0:
            #     print(percentage_of_token_in_range, token, feature_relevant_range)
            tokens_with_scores.append({"word": token["word_idx"], "score": int(percentage_of_token_in_range > 0.9)})
    
    train_data_preprocessed[i] = {
                                    "pn_history_tokens": torch.tensor([ts["word"] for ts in tokens_with_scores]),
                                    "scores": torch.tensor([ts["score"] for ts in tokens_with_scores]),
                                    "feature_tokens": torch.tensor(tokenized_features[feature_text])
                                   }
        

# %%
num_no_positives = sum([1 for dp in train_data_preprocessed.values() if sum(dp["scores"]) == 0])
print(f"filtering {num_no_positives} out of {len(train_data_preprocessed)} datapoints because they don't contain any positive scores.")
train_data_preprocessed = {key: dp for key, dp in train_data_preprocessed.items() if sum(dp["scores"]) != 0}

# %%
train_data_preprocessed[0]

# %% [markdown]
# # TODO Bring feature label into training data for LSTM!
# - must encode the feature text into the LSTM input data to train. How to do it?
# - 2 vocabs
# 
# Layers in LSTM Model:
# 1. embed feature tokens
# 2. lstm feature -> constant size vector
# 
# 3. pass to 2nd lstm
# 	

# %%
EMBEDDING_DIM = 128
HIDDEN_DIM = 256


# %%
class LSTMTokenScorer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pn_hist_vocab_size, feature_vocab_size, dropout=0.0):
        super(LSTMTokenScorer, self).__init__()

        self.pn_history_hidden_dim = hidden_dim

        self.feature_embeddings = nn.Embedding(feature_vocab_size, embedding_dim)
        self.feature_lstm = nn.LSTM(embedding_dim, embedding_dim, dropout=dropout) # the feature is now one tensor of size [embedding_dim].

        self.pn_history_embeddings = nn.Embedding(pn_hist_vocab_size, embedding_dim)
        
        self.total_lstm = nn.LSTM(embedding_dim * 2, self.pn_history_hidden_dim, dropout=dropout)
        
        self.hidden2score = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, pn_history, feature):
        feature_embeds = self.feature_embeddings(feature)
        feature_lstm_out, _ = self.feature_lstm(feature_embeds.view(len(feature), 1, -1)) # the feature is now one tensor of size [embedding_dim].
        feature_reduced = torch.squeeze(feature_lstm_out[-1]) #.view(1, -1)
        feature_multiplied = feature_reduced.repeat((len(pn_history), 1)) # duplicate feature vector to be same size as embedded pn_history vector.

        pn_history_embeds = self.pn_history_embeddings(pn_history)
        pn_history_and_features = torch.concat((feature_multiplied, pn_history_embeds), dim=1)

        pn_history_reduced, _ = self.total_lstm(pn_history_and_features)
        pred_score_raw = torch.squeeze(self.hidden2score(pn_history_reduced))
        pred_score = self.sigmoid(pred_score_raw)
        return pred_score

# %%
all_scores = [d["scores"].numpy() for d in train_data_preprocessed.values()]
avg_neg_div_pos = np.mean([(scores.shape[0] - np.sum(scores)) / np.sum(scores) for scores in all_scores])

# %%
# make model with vocab sizes, including placeholder indices
model = LSTMTokenScorer(EMBEDDING_DIM, HIDDEN_DIM, len(pn_history_vocab)+1, len(feature_vocab)+1)
loss_function = nn.BCELoss()
# loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(avg_neg_div_pos))
# loss_function = lambda pred, target, vec_size: nn.functional.binary_cross_entropy_with_logits(pred.float(), target.float(), pos_weight=torch.full((vec_size,), avg_neg_div_pos))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %%
feature_tokens = train_data_preprocessed[0]["feature_tokens"]
pn_history_tokens = train_data_preprocessed[0]["pn_history_tokens"]

model(pn_history_tokens, feature_tokens)
# model()

# %%
# def one_hot_encode(val):
#     if val == 0:
#         return torch.tensor([1, 0], dtype=torch.float)
#     elif val == 1:
#         return torch.tensor([0, 1], dtype=torch.float)
#     raise Exception("one hot encode got invalid value.")

# %%
import random
logfile_name = "training_log.txt"

def log(logtext: str = "") -> None:
    print(logtext)
    with open(logfile_name, "a", encoding="utf8") as f:
        f.write(str(logtext) + "\n")
    

def train_model(model: LSTMTokenScorer, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = 9999999999999

        for epoch in range(num_epochs):
            log(f'Epoch {epoch}/{num_epochs - 1}')
            log('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train']: #, 'test'
                if phase == 'train':
                    model.train()
                else: 
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0

                batch = random.choices(list(train_data_preprocessed.values()), k=64)

                # Iterate over data.
                for i, datum_preprocessed in enumerate(batch):
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        pn_history_tokens = datum_preprocessed["pn_history_tokens"]
                        scores = datum_preprocessed["scores"]
                        feature_tokens = datum_preprocessed["feature_tokens"]
                        
                        outputs = model(pn_history_tokens, feature_tokens)
                        loss = criterion(outputs.float(), scores.float())

                        pred = (outputs > 0.9).int()
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    if torch.equal(pred, scores):
                        running_corrects += 1
                    
                    if i == len(batch) - 1 and epoch % 20 == 19:
                        log("LSTM output:")
                        log(outputs)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss # / dataset_sizes[phase]
                epoch_acc = running_corrects # / dataset_sizes[phase]
                log(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time elapsed: {round((time.time() - since))} sec.')
                
                # deep copy the model
                if phase == 'test' and epoch_loss < best_loss: #epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            log()

        time_elapsed = time.time() - since
        log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        log(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

open(logfile_name, "w", encoding="utf8")
    
model = train_model(model, loss_function, optimizer, exp_lr_scheduler, num_epochs=100)

# %%
a = torch.tensor([0, 0, 0], dtype=torch.float)
b = torch.tensor([0, 0, 1], dtype=torch.float)
c = torch.tensor([1, 1, 0], dtype=torch.float)
d = torch.tensor([0, 1, 0], dtype=torch.float)

nn.functional.binary_cross_entropy_with_logits(a, b, pos_weight=torch.full((3,), 2))


