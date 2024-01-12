# CNN
import os
import time
from tempfile import TemporaryDirectory
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import Iterable
from transformers import BertTokenizerFast
from os.path import join as pathjoin
import pickle
from pytorch_pretrained_bert.modeling import BertModel
from functools import lru_cache
from typing import Dict, List
import random
from sklearn.metrics import f1_score

root = '../data/'
features_path = os.path.join(root, 'features.csv')
patient_notes_path = os.path.join(root, 'patient_notes.csv')
sample_submission_path = os.path.join(root, 'sample_submission.csv')
test_path = os.path.join(root, 'test.csv')
train_path = os.path.join(root, 'train.csv')
features = pd.read_csv(features_path, sep=',', header=0)
patient_notes = pd.read_csv(patient_notes_path, sep=',', header=0)
train_raw = pd.read_csv(train_path, sep=',', header=0)



# data preprocessing: Convert character ranges from strings to tuples of integers
def df_string2list_of_ints(df_string: str):
    df_string = df_string.strip("[]")
    if df_string == "":
        return []
    entries = re.split(",|;", df_string)
    entries = [entry.strip(" '") for entry in entries]
    ranges = [tuple(int(num_as_str) for num_as_str in entry.split(" ")) for entry in entries]
    return ranges


data_merged = train_raw.merge(features, on=['feature_num', 'case_num'], how='left')
data_merged = data_merged.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
data_merged["location"] = data_merged["location"].apply(df_string2list_of_ints)
data_merged.head()

train = data_merged[["feature_text", "pn_history", "location", ]]
train.head()

# filter out training data with no location
train = train[train["location"].apply(lambda row: len(row) != 0)]

print(f'Size of dataset= {len(train)}')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
cache_dir = "cache"

# get bert embedding function (matrix)

BERT_FP = ('bert-base-uncased')


def get_bert_embed_matrix():
    bert = BertModel.from_pretrained(BERT_FP)
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat


embedding_matrix = get_bert_embed_matrix()



def embed_seq(s: Iterable[int]):
    return np.array([onehot_word(word_id) @ embedding_matrix for word_id in s])


@lru_cache(maxsize=10000)
def embed_word(word_id: int):
    return onehot_word(word_id) @ embedding_matrix


def onehot_word(a: int):
    oh = np.zeros(30522, dtype=int)
    oh[a] = 1
    return oh


# tokenize patient histories

cache_file = pathjoin(cache_dir, "tokenized_pn_histories.pkl")
if os.path.isfile(cache_file):
    with open(cache_file, "rb") as f:
        tokenized_pn_histories = pickle.load(f)
    print("Tokenized patient histories loaded from cache.")
else:
    print("Found no cached tokenized patient histories. Tokenizing...")
    tokenized_pn_histories: Dict[str, List[Dict]] = {}
    for pn_history in tqdm(train["pn_history"]):
        indexed_words = []
        if pn_history in tokenized_pn_histories:
            continue

        tokenized = tokenizer.encode_plus(pn_history, return_offsets_mapping=True, add_special_tokens=True)

        for word, offset_mapping in zip(tokenized["input_ids"], tokenized["offset_mapping"]):
            embedded_token = embed_word(word)

            indexed_words.append({
                "word_id": word,
                "embedded": embedded_token,
                "range": offset_mapping
            })

        tokenized_pn_histories[pn_history] = indexed_words
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_pn_histories, f)

# Data structure:
# tokenized_pn_histories
# hist_id -> [tokens]
# token -> ['word_id', 'embedded', 'range']
#



# tokenize features
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

        tokenized = tokenizer.encode_plus(feature_text, add_special_tokens=True)

        for word in tokenized["input_ids"]:
            embedded_token = embed_word(word)

            indexed_words.append({
                "word_id": word,
                "embedded": embedded_token,
            })

        tokenized_features[feature_text] = indexed_words
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_features, f)

# final preparation of training data: merge patient histories with features, with scores [0 or 1]
# indicating whether a token is part of the character range describing the feature.

train_data_preprocessed = dict()
for i, (feature_text, pn_history, location) in tqdm(train.iterrows()):
    tokenized_history = tokenized_pn_histories[pn_history]
    tokens_with_scores = []
    for token in tokenized_history:
        percentages = []
        for feature_relevant_range in location:
            token_start, token_end = token["range"]
            range_start, range_end = feature_relevant_range[0], feature_relevant_range[1]

            percentage_of_token_in_range = max(min(token_end, range_end) + 1 - max(token_start, range_start), 0) / (
                    token_end + 1 - token_start)
            percentages.append(percentage_of_token_in_range)

        tokens_with_scores.append({"token": token,
                                   "score": int(max(percentages) > 0.9)})

    train_data_preprocessed[i] = {
        "pn_history_tokens": [ts["token"] for ts in tokens_with_scores],
        "scores": torch.tensor([ts["score"] for ts in tokens_with_scores]),
        "feature_tokens": tokenized_features[feature_text],
        "locations": location
    }

num_no_positives = sum([1 for dp in train_data_preprocessed.values() if sum(dp["scores"]) == 0])
print(
    f"filtering {num_no_positives} out of {len(train_data_preprocessed)} datapoints because they don't contain any positive scores.")
train_data_preprocessed = {key: dp for key, dp in train_data_preprocessed.items() if sum(dp["scores"]) != 0}

# %% [markdown]
# # Structure of the Model
# Layers in LSTM Model:
# 1. embed feature tokens
# 2. lstm feature -> constant size vector
# 3. pass to 2nd lstm
# 	

EMBEDDING_DIM = embedding_matrix.shape[1]
HIDDEN_DIM = 256


class LSTMTokenScorer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super(LSTMTokenScorer, self).__init__()

        self.pn_history_hidden_dim = hidden_dim
        self.feature_lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=False,
                                    dropout=dropout)  # the feature is now one tensor of size [embedding_dim].
        self.total_lstm = nn.LSTM(embedding_dim * 2, self.pn_history_hidden_dim, bidirectional=False, dropout=dropout)
        self.hidden2score = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pn_history, feature):
        feature_lstm_out, _ = self.feature_lstm(
            feature.view(len(feature), 1, -1))  # the feature is now one tensor of size [embedding_dim].
        feature_reduced = torch.squeeze(feature_lstm_out[-1])  # .view(1, -1)
        feature_multiplied = feature_reduced.repeat(
            (len(pn_history), 1))  # duplicate feature vector to be same size as embedded pn_history vector.
        pn_history_and_features = torch.concat((feature_multiplied, pn_history), dim=1)

        pn_history_reduced, _ = self.total_lstm(pn_history_and_features)
        pred_score_raw = torch.squeeze(self.hidden2score(pn_history_reduced))
        pred_score = self.sigmoid(pred_score_raw)
        return pred_score



logfile_name = "training_log.txt"


def log(logtext: str = "") -> None:
    print(logtext)
    with open(logfile_name, "a", encoding="utf8") as f:
        f.write(str(logtext) + "\n")


def train_model(model: LSTMTokenScorer, criterion, optimizer, scheduler, num_epochs=1, prev_losses=[], prev_f1s=[]):
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_f1 = 0.0
        best_loss = 9999999999999

        losses = prev_losses
        f1s = prev_f1s
        for epoch in range(num_epochs):
            log(f'Epoch {epoch}/{num_epochs - 1}')
            log('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_loss_average = 0.0
                running_f1_total = 0.0
                running_f1_average = 0.0
                num_non_zero_outputs_in_epoch = 0.0

                # batch = random.choices(list(train_data_preprocessed.values()), k=64)
                data_ids = list(train_data_preprocessed.keys())
                random.shuffle(data_ids)

                # Iterate over data.
                for i, data_id in enumerate(data_ids):
                    datum_preprocessed = train_data_preprocessed[data_id]
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    pn_history_tokens = datum_preprocessed["pn_history_tokens"]
                    scores = datum_preprocessed["scores"]
                    feature_tokens = datum_preprocessed["feature_tokens"]

                    feature_tensor = torch.tensor(np.array([t["embedded"] for t in feature_tokens]), dtype=torch.float)
                    pn_history_tensor = torch.tensor(np.array([t["embedded"] for t in pn_history_tokens]),
                                                     dtype=torch.float)

                    # track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(pn_history_tensor, feature_tensor)

                        num_non_zero_outputs = np.count_nonzero(outputs.detach().numpy().round().astype(int))
                        num_non_zero_outputs_in_epoch += num_non_zero_outputs

                        loss = criterion(outputs.float(), scores.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    try:
                        f1 = f1_score(scores.int(), outputs.detach().round().int())
                    except Exception as e:
                        log("F1 score calc failed:")
                        log("Scores:")
                        log(scores.int())
                        log("\nOutputs")
                        log(outputs.detach().round().int())
                        log("\n")
                        raise Exception(e)
                    # statistics
                    running_loss += loss.item()
                    running_loss_average = running_loss / (i + 1)
                    losses.append(loss.item())
                    running_f1_total += f1
                    running_f1_average = running_f1_total / (i + 1)
                    f1s.append(f1)

                    if i % 1000 == 0:
                        # output the nonzero outputs too. At the beginning of training, it is common for all the outputs to be zero (rounded).
                        log(f"Epoch {epoch}, i={i}, avg. loss={running_loss_average}, avg. F1={running_f1_average}, nonzero outputs in epoch={num_non_zero_outputs_in_epoch}")

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss
                epoch_f1 = running_f1_average
                log(f'{phase} Loss: {epoch_loss:.4f} F1: {epoch_f1:.4f} Time elapsed: {round((time.time() - since))} sec.')

                # deep copy the model
                if phase == 'test' and epoch_loss < best_loss:
                    best_f1 = epoch_f1
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            log()

        time_elapsed = time.time() - since
        log(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        log(f'Best val Acc: {best_f1:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, losses, f1s


data_ids = list(train_data_preprocessed.keys())
neg_values = 0.0
pos_values = 0.0
for i, data_id in tqdm(enumerate(data_ids)):
    datum_preprocessed = train_data_preprocessed[data_id]

    pn_history_tokens = datum_preprocessed["pn_history_tokens"]
    scores = datum_preprocessed["scores"]
    pos_values += np.count_nonzero(scores.numpy().round().astype(int))
    neg_values += (scores.shape[0] - np.count_nonzero(scores.numpy().round().astype(int)))

neg_pos_ratio = neg_values / pos_values

open(logfile_name, "w", encoding="utf8")  # clear logs

lr = 0.001
model = LSTMTokenScorer(EMBEDDING_DIM, HIDDEN_DIM)
pos_weight = torch.full((1,), neg_pos_ratio)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # make positive class more valuable

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

log(f"Starting model training for lr={lr}...")
model, losses, f1s = train_model(model, loss_function, optimizer, exp_lr_scheduler, num_epochs=20)

with open(pathjoin(cache_dir, "losses2.pkl"), "wb") as f:
    pickle.dump(losses, f)
with open(pathjoin(cache_dir, "f1s2.pkl"), "wb") as f:
    pickle.dump(f1s, f)


def smooth(x: Iterable, N: int = 10_000):
    return np.convolve(x, np.ones(N) / N, mode='valid')


# plot smoothed f1 score over time
plt.plot(smooth(f1s))
