#!/usr/bin/env python
"""
Project: ML pipeline for Shot-Term Rental Prices in NYC
Author: quockhanh0212
Date: 2023-10-09
"""

import os
import json
import tempfile

import mlflow

import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model"
]

# This automatically reads in the configuration


@hydra.main(config_name='config')
def go(config: DictConfig):
    """
    This function executes a series of steps based on the provided configuration.

    Parameters:
    config (DictConfig): A configuration dictionary containing the following keys:
        - main: Contains project and experiment names, and steps to execute.
        - etl: Contains parameters for the ETL process.
        - data_check: Contains parameters for the data check process.
        - modeling: Contains parameters for the modeling process.

    The function executes the following steps:
    1. Setup the wandb experiment.
    2. Download file and load in W&B (if 'download' is in active_steps).
    3. Perform basic data cleaning (if 'basic_cleaning' is in active_steps).
    4. Perform data check (if 'data_check' is in active_steps).
    5. Split data into training and testing sets (if 'data_split' is in active_steps).
    6. Train a random forest model (if 'train_random_forest' is in active_steps).
    7. Test the regression model (if 'test_regression_model' is in active_steps).

    Each step is executed using MLflow with specific parameters defined in the config.

    """

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                uri=os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "basic_cleaning"),
                entry_point="main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']},
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(),
                                 "src", "data_check"),
                entry_point="main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                uri=f"{config['main']['components_repository']}/train_val_test_split",
                entry_point="main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "random_seed": config['modeling']['random_seed']},
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(
                    dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                uri=os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "train_random_forest"),
                entry_point="main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": "random_forest_model",
                },
            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                uri=f"{config['main']['components_repository']}/test_regression_model",
                entry_point="main",
                parameters={
                    "mlflow_model": "random_forest_model:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
