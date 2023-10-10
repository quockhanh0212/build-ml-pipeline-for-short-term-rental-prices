# Build an ML Pipeline for Short-term Rental Prices in NYC

[**Project Description**](#project-description) | [**Install**](#install) | [**Login to Wandb**](#login-to-wandb) | [**Cookiecutter**](#cookiecutter) | [**Hydra**](#hydra) | [**Pandas Profiling**](#pandas-profiling) | [**Release new version**](#release-new-version) | [**Step-by-step**](#step-by-step) | [**Public Wandb project**](#public-wandb-project) | [**Code Quality**](#code-quality)

## Project Description
Working for a property management company renting rooms and properties for short periods of time on various platforms. Need to estimate the typical price for a given property based on the price of similar properties. Your company receives new data in bulk every week. The model needs to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

## Install
In order to run these components you need to have conda (Miniconda or Anaconda) and MLflow installed.
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

## Login to Wandb
```bash
wandb login
```

## Cookiecutter
Using this template you can quickly generate new steps to be used with MLFlow.
```bash
cookiecutter cookiecutter-mlflow-template -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

## Hydra
As usual, the parameters controlling the pipeline are defined in the config.yaml file defined in the root of the starter kit. We will use Hydra to manage this configuration file. Open this file and get familiar with its content. Remember: this file is only read by the main.py script (i.e., the pipeline) and its content is available with the go function in main.py as the config dictionary. For example, the name of the project is contained in the project_name key under the main section in the configuration file. It can be accessed from the go function as config["main"]["project_name"].

## Pandas Profiling
ydata-profiling primary goal is to provide a one-line Exploratory Data Analysis (EDA) experience in a consistent and fast solution. Like pandas df.describe() function, that is so handy, ydata-profiling delivers an extended analysis of a DataFrame while allowing the data analysis to be exported in different formats such as html and json.
```python
pip install ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Profiling Report")
profile.to_widgets()
```

## Release new version
```bash
git tag -a 1.0.1 -m "Release 1.0.1"
git push origin 1.0.1
```

## Step-by-step
### 0. Full pipeline
```bash
mlflow run .
```

### 1. Download data
```bash
mlflow run . -P steps=download
```

### 2. EDA
```bash
mlflow run src/eda
```
More details in [![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F.svg?style=for-the-badge&logo=jupyter&logoColor=white)](src/eda/EDA.ipynb)



### 3. Basic cleaning
```bash
mlflow run . -P steps=basic_cleaning
...
2023-10-09 09:52:46,799 Dropping duplicates
2023-10-09 09:52:46,917 Dropping outliers
2023-10-09 09:52:46,917 Number of rows before dropping outliers: 48895
2023-10-09 09:52:46,953 Number of rows after dropping outliers: 46428
2023-10-09 09:52:46,953 Converting last_review to datetime
2023-10-09 09:52:46,963 Saving cleaned dataframe to csv
2023-10-09 09:52:47,287 Logging artifact
```

### 4. Check data
```bash
mlflow run . -P steps=check_data
...
test_data.py::test_column_names PASSED                                   [ 16%]
test_data.py::test_neighborhood_names PASSED                             [ 33%]
test_data.py::test_proper_boundaries PASSED                              [ 50%]
test_data.py::test_similar_neigh_distrib PASSED                          [ 66%]
test_data.py::test_price_range PASSED                                    [ 83%]
test_data.py::test_row_count PASSED                                      [100%]
```

### 5. Split data
```bash
mlflow run . -P steps=data_split
...
2023-10-08 03:23:27,440 Fetching artifact clean_sample.csv:latest
2023-10-08 03:23:32,517 Splitting trainval and test
2023-10-08 03:23:32,597 Uploading trainval_data.csv dataset
2023-10-08 03:23:40,856 Uploading test_data.csv dataset
```

### 6. Train and evaluate model
```bash
mlflow run . -P steps=train_random_forest
...
2023-10-08 16:36:05,008 Minimum price: 10, Maximum price: 350
2023-10-08 16:36:05,032 Preparing sklearn pipeline
2023-10-08 16:36:05,034 Fitting
2023-10-08 16:37:37,819 Computing and scoring r2 and MAE
2023-10-08 16:37:37,946 Score: 0.5554115487254865
2023-10-08 16:37:37,946 MAE: 33.24032730263158
2023-10-08 16:37:37,946 Exporting model
```

Optimize hyper-parameters
```bash
mlflow run . \
   -P steps=train_random_forest \
   -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```
![hyper_parameters](/projects/reproducible_model_workflow/images/optimize_hyper_parameters.png)

### 7. Test model
```bash
mlflow run . -P steps=test_regression_model
...
2023-10-08 17:29:01,005 Downloading artifacts
2023-10-08 17:29:06,837 Loading model and performing inference on test set
2023-10-08 17:29:07,055 Scoring
2023-10-08 17:29:07,138 Score: 0.5735931364721438
2023-10-08 17:29:07,138 MAE: 32.719363325440675
```

### 8. Test model with new dataset
```
mlflow run https://github.com/quockhanh0212/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"
...
2023-10-09 10:02:13,629 Dropping duplicates
2023-10-09 10:02:13,683 Dropping outliers
2023-10-09 10:02:13,684 Number of rows before dropping outliers: 48895
2023-10-09 10:02:13,704 Number of rows after dropping outliers: 46427
2023-10-09 10:02:13,705 Converting last_review to datetime
2023-10-09 10:02:13,715 Saving cleaned dataframe to csv
2023-10-09 10:02:14,113 Logging artifact
...
test_data.py::test_column_names PASSED                            [ 16%]
test_data.py::test_neighborhood_names PASSED                      [ 33%]
test_data.py::test_proper_boundaries PASSED                       [ 50%]
test_data.py::test_similar_neigh_distrib PASSED                   [ 66%]
test_data.py::test_price_range PASSED                             [ 83%]
test_data.py::test_row_count PASSED                               [100%]
...
2023-10-09 10:03:22,584 Fetching artifact clean_sample.csv:latest
2023-10-09 10:03:24,892 Splitting trainval and test
2023-10-09 10:03:24,965 Uploading trainval_data.csv dataset
2023-10-09 10:03:32,192 Uploading test_data.csv dataset
...
## Public Wandb project
Link: https://wandb.ai/quockhanh0212/nyc_airbnb/overview?workspace=user-quockhanh0212

Select best model
![wandb-select-best](https://video.udacity-data.com/topher/2021/March/605103d6_wandb-select-best/wandb-select-best.gif)

## Code Quality
Style Guide - Format your refactored code using PEP 8 – Style Guide. Running the command below can assist with formatting. To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below:
```bash
autopep8 --in-place --aggressive --aggressive .
```

Style Checking and Error Spotting - Use Pylint for the code analysis looking for programming errors, and scope for further refactoring. You should check the pylint score using the command below.
```bash
pylint -rn -sn .
```
Docstring - All functions and files should have document strings that correctly identifies the inputs, outputs, and purpose of the function. All files have a document string that identifies the purpose of the file, the author, and the date the file was created.