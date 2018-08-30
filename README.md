# Kickstarter Model and Prediction

## Description

The goal of this project is to build a predictive machine learning model to predict a Kickstarter project's state (successful or unsuccessful). This repository consists of two parts - a jupyter notebook with comprehensive code written for data cleaning, preprocessing, exploratory data analysis and visualization, modeling, evaluation, and hyperparameter tuning, and a python module to serialize the trained machine learning model and expose it as an API for users/ engineers.

## Getting Started

This project was built on python 3.6 version. Ensure you have python 3.6 version installed and running on your machine and then follow the below instructions:

1. Clone the project from the repository
2. Run the makefile to create a virtual environment and load all the dependencies
3. Run training.py file at 'model/' path. This will take 2-3 minutes approximately. Once this file runs successfully, it will create a serialized trained model named 'model_v1.pk' at 'api/models/ path
4. Run apicall.py file at 'api/' path. Once the host is up and running, use the API to test predictions. (Note - For illustration, I have included a test_json.json and used it to make the first prediction. You can comment that code and pass your own json object via 'POST' and get the predictions)
5. The jupyter notebook is also checked in along with the python module at 'model/' path
