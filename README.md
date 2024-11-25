# Mind-Matters: Exploring Mental Health Data

## Overview

This project is part of Kaggle's Playground Season 4, Episode 11 Competition, focused on analyzing mental health data
with an emphasis on understanding and predicting factors related to depression. The competition provides a structured
dataset that allows for exploratory data analysis (EDA), feature engineering, and predictive model development to tackle
the challenges of mental health research. By leveraging this dataset, the project aims to uncover meaningful insights
and build robust models to contribute to a deeper understanding of depression and its associated factors, aligning with
the goals of the Kaggle competition

## Project Structure

The project is organized as follows:

```bash
mind-matters/
├── data/                 # Contains raw and processed datasets            
├── models/               # Saves trained models
├── notebooks/            # Jupyter notebooks for experimentation and analysis
│   ├── 01_univariate_exploratory_data_analysis.ipynb
│   ├── 02_bivariate_exploratory_data_analysis.ipynb
│   └── 03_feature_engineering.ipynb
├── src/                  # Source code for the project
│   ├── data/             # Scripts for data preprocessing
│   ├── pipeline/         # Pipelines for modeling and evaluation
│   ├── transformers/     # Custom transformers for preprocessing
│   └── validation/       # Validation utilities
├── utils/                # Helper functions and utilities
│   ├── data.py           # Functions for data handling
│   ├── helpers.py        # General helper functions
│   ├── logger.py         # Logging utilities
│   └── plot.py           # Visualization utilities
├── .gitignore            # Specifies files to ignore in version control
└── main.py               # Entry point for running the project
```

## Installation

### Option 1: Python Environment

1. Clone the repository:

```bash
git clone https://github.com/ktxdev/mind-matters.git
cd mind-matters
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate 
```

3. Install dependencies:

```bash
pip install -r requirements.txt 
```

### Option 2: Conda environment

1. Clone the repository:

```bash
git clone https://github.com/ktxdev/mind-matters.git
cd mind-matters
```

2. Create and activate a Conda environment:

```bash
conda create --name mind-matters python=3.9 -y
conda activate mind-matters 
```

3. Install dependencies:

```bash
pip install -r requirements.txt 
```

3. Set up Jupyter Notebook (optional): If you're planning to run the notebooks:

```bash
conda install -c conda-forge notebook 
```

## Usage

### 1. Data Exploration

The notebooks/ directory contains Jupyter notebooks for EDA:

- `01_univariate_exploratory_data_analysis.ipynb`: Univariate analysis of features.
- `02_bivariate_exploratory_data_analysis.ipynb`: Analyzing relationships between variables.
- `03_feature_engineering.ipynb`: Transforming features for better modeling.

### 2. Running the Project

To run the main pipeline, execute:

```bash
python main.py --train xgb
```

## Features

- **EDA:** Thorough exploration of univariate and bivariate distributions.
- **Feature Engineering:** Handling missing data, scaling, encoding, and creating new features.
- **Modeling:** Building machine learning models with libraries like CatBoost.
- **Visualization:** Plots and charts for better understanding of the data.

## Technologies Used

<div style="display: flex; justify-content: center; align-items: center; gap: 20px; flex-wrap: wrap;">
<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" style="width: auto; height: 60px; object-fit: contain;"/>
<img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="Pandas" style="width: auto; height: 60px; object-fit: contain;"/> 
<img src="https://numpy.org/images/logo.svg" alt="NumPy" style="width: auto; height: 60px; object-fit: contain;"/> 
<img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Scikit-learn" style="width: auto; height: 60px; object-fit: contain;"/> 
<img src="https://jupyter.org/assets/homepage/main-logo.svg" alt="Jupyter" style="width: auto; height: 60px; object-fit: contain;"/> 
<img src="https://matplotlib.org/_static/logo_light.svg" alt="Matplotlib" style="width: auto; height: 60px; object-fit: contain;"/> 
<img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" alt="Seaborn" style="width: auto; height: 60px; object-fit: contain;"/> 
</div>

## Acknowledgements 
- **Data source:** [Kaggle: Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11/data)