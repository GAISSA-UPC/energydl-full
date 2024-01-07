# energydl-full

This is the replication package to the publication .... To reproduce the study, do as follows:

## Install requirements

`pip install -r requirements.txt`

## Generate data

`python3 src/data/generate_train_data.py` 

Take into account that this consumes lots of resources and you might need from a GPU to execute this script. It should generate the files `monitor.csv`, `history.csv`, `emissions.csv` similar to those in `data/raw`. Make sure you have also `data/raw/datasets.csv`, which was manually created.

`Rscript preprocess.R`

This should generate the file called `all_data.csv` similar to the one in `data/processed`.

## Analyze the data

Open with Rstudio the files `src/analysis/analysis_rq1.Rmd` and `src/analysis/analysis_rq2.Rmd` and execute all the cells in order to get insights on the data, generate the figures seen in the study and answer research questions 1 and 2 respectively.

## Execute the application

This part is out of the scope from the study, although it is left as a proof of concept for future directions of the results of this work. An application of the results has been developed on the form of a Python library, available in `src/app/energydl.py`. A demo of its usage can be tried by calling: `python3 src/app/demo_energydl.py` in a machine with a GPU. 
