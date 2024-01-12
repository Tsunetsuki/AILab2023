# Introduction 
The relevent models can be found in the following files:
- lstm/
  - [lstm-with-bert.py](lstm/lstm-with-bert.py)
- deberta/
  - [nmbe-bert-train.ipynb](deberta/nmbe-bert-train.ipynb)
  - [nmbe-bert-train.py](deberta/nmbe-bert-train.py)
  - [nmbe-deberta-train.ipynb](deberta/nmbe-deberta-train.ipynb)
  - [nmbe-deberta-transferlearn.ipynb](deberta/nmbe-deberta-transferlearn.ipynb)

Local copies of the data provided by Kaggle can be found in the "data" folder
Other Python and Jupyter files were used for preparation steps like data exploration.

# File Structure 
```bash 
❯ tree -L 2 --dirsfirst\

.
├── data # datasource from Kaggle 
│   ├── README.md
│   ├── features.csv
│   ├── patient_notes.csv
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── deberta # all deberta related files 
│   ├── tokenizer
│   ├── README.md
│   ├── config.pth
│   ├── nmbe-bert-train.ipynb
│   ├── nmbe-bert-train.py
│   ├── nmbe-deberta-train copy.py
│   ├── nmbe-deberta-train.ipynb
│   ├── nmbe-deberta-train.py
│   ├── nmbe-deberta-transferlearn.ipynb
│   └── nmbe-deberta-transferlearn.py
├── lstm # all lstm related files 
│   ├── lstm-with-bert.py
├── README.md
├── data-view.ipynb # data exploration and visualization 
└── resources.ipynb # the detail of the resources used in this project 

5 directories, 20 files
```