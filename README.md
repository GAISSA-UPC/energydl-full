# energydl-full

This is the replication package to the publication .... The execution follows 3 steps:

## Generate data

Calling `src/data/generate_train_data.py` should generate the files `monitor.csv`, `history.csv`, `emissions.csv` similar to those in `data/raw`. Make sure you have also `data/raw/datasets.csv`, which was manually created. Then, executing all the cells in order from `src/data/preprocess.Rmd` should create one only file called `all_data.csv` similar to the one in `data/processed`.

## Analyze the data

Execute all the cells in order from `src/analysis/analysis_rq1.Rmd` and `src/analysis/analysis_rq2.Rmd` to get insights on the data, generate the figures seen in the study and answer research questions 1 and 2 respectively.

## Execute the application

This part is out of the scope from the study, although it is left as a proof of concept for future directions of the results of this work. An application of the results has been developed on the form of a Python library, available in `src/app/energydl.py`. A demo of its usage can be found on `src/app/demo_energydl.py`. This demo was shown to a group of volunteers who then answered the survey in `documents/questionaire.txt`. Their answers can be found in `data/raw/form_answers.csv`, and can be analyzed by executing all cells in order from `src/analysis/analysis_rq2.Rmd`.
