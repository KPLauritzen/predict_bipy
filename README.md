predict_bipy
==============================

Using recurrent neural networks to classify conductance traces.
Code repository for the "molecular trace detection" part of the paper
"Classification of conductance traces with recurrent neural networks" by Kasper
Lauritzen, András Magyarkuti, Zoltán Balogh, András Halbritter, and Gemma C.
Solomon (Accepted JCP 2018) 

Usage
-----
First, raw conductance traces should be places in `data/external` folder.
- 4K traces are places in folders (BP_FlipsOut/ BP_Stays/ Tunnel/ UnSorted/)
- 300K traces are placed in (BP_300K/)

Then the data is processed with `src/data/make_4K_dataset.py`.
This will be the input to the neural network. 

The network is trained with `src/models/train_predict.py`
This script also outputs the predicted probabilities from the RNN

The performance of the network is evaulated with Jupyter notebooks and also
`src/models/training_log_analysis.py`.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from Halbritter group
    │   ├── interim        <- Not used
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- Not used
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
