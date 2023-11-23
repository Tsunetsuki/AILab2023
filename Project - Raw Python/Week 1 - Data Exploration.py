import pandas as pd
from collections import Counter
import os
import re

pd.set_option('display.width', 1000)

root = './data/'
features_path = os.path.join(root, 'features.csv')
patient_notes_path = os.path.join(root, 'patient_notes.csv')
sample_submission_path = os.path.join(root, 'sample_submission.csv')
test_path = os.path.join(root, 'test.csv')
train_path = os.path.join(root, 'train.csv')


features = pd.read_csv(features_path, sep=',', header=0)
print("\nFeatures:")
print(features)

patient_notes = pd.read_csv(patient_notes_path, sep=',', header=0)
print("\nPatient Notes:")
print(patient_notes)

sample_submission = pd.read_csv(sample_submission_path, sep=',', header=0)
print("\nSample submission:")
print(sample_submission)


def df_string2list_of_ints(df_string: str):
    df_string = df_string.strip("[]")
    if df_string == "":
        return []
    entries = re.split(",|;", df_string)
    entries = [entry.strip(" '") for entry in entries]
    ranges = [tuple(int(num_as_str) for num_as_str in entry.split(" ")) for entry in entries]
    return ranges

print("\nTrain data (unmodified):")
print(pd.read_csv(train_path, sep=',', header=0))

train = pd.read_csv(train_path, sep=',', header=0)
train = train.merge(features, on=['feature_num', 'case_num'], how='left')
train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
train["location"] = train["location"].apply(df_string2list_of_ints)
print("\nTrain data (merged with features + patient notes):")
print(train)

num_rows_without_location = train["location"].apply(lambda row: len(row) == 0).sum() / len(train)
print(f"\nPercentage of objects in training data without locations: {round(100 * num_rows_without_location)}%")
print("  Examples:")
print(train[train["location"].apply(lambda row: len(row) == 0)])