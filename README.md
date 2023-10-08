# MLP Regressor Winrate Calculation

This Python script calculates the winrate of a Multi-Layer Perceptron (MLP) Regressor model using a given dataset. The MLP Regressor is trained to predict a target variable based on input features. The winrate is determined by comparing the predicted values with the actual values and calculating the ratio of correct predictions to the total number of predictions.

## Prerequisites

- Python
- scikit-learn

## Usage

1. Clone this repository or download the script.
2. Install the required Python packages using `pip install scikit-learn pandas numpy`.
3. Modify the path to your dataset in the script (`DataForNN.xlsx`).
4. Run the script using `python script_name.py`, where `script_name.py` is the name of your script.

## Description

This script performs the following tasks:

- Loads a dataset from an Excel file.
- Preprocesses the data, including normalization of the target variable.
- Splits the data into training and testing sets.
- Creates an MLP Regressor model with specified hyperparameters.
- Trains the model on the training data.
- Makes predictions on the test data.
- Calculates the winrate based on the correctness of predictions.
- The winrate represents the accuracy of the model in predicting the target variable.

## Customization

You can customize the script by adjusting hyperparameters, changing the dataset path, or modifying the input features and target variable columns.

## License

This project is licensed under the [MIT License](LICENSE).
