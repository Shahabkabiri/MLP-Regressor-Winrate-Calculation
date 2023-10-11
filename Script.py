import pandas as pd
import numpy as np  # Import NumPy for array operations

# Load the data from an Excel file
DataForNN = pd.read_excel('DataForNN.xlsx')

# Define a function for MLP Regressor Winrate Calculation
def MLPRegressorWinrateCalculation(DataForNN):
    from sklearn.neural_network import MLPRegressor  # Import MLP Regressor
    from sklearn.model_selection import train_test_split  # Import train-test split
    from sklearn.preprocessing import MinMaxScaler  # Import Min-Max scaler
    from sklearn.metrics import confusion_matrix as cm  # Import confusion matrix

    # Check if the data size is greater than 20
    if DataForNN.shape[0] > 20:
        print('Database Size:', DataForNN.shape[0])
        X = np.array(DataForNN[['Total Volume', 'Price Change']])  # Input features
        Y = np.array(DataForNN[['Successful Order']]).ravel()  # Target variable

        # Normalize the target variable
        normalized_Y = (Y - Y.min()) / (Y.max() - Y.min())
        print(Y)

        # Split the data into training and testing sets
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=.3, random_state=1)

        # Create an MLP Regressor model
        model = MLPRegressor(hidden_layer_sizes=(80, 200, 80), activation='logistic',
                             solver='sgd', batch_size=5, learning_rate='adaptive',
                             learning_rate_init=0.01, max_iter=300, shuffle=True, tol=.00000001, verbose=False,
                             momentum=0.5)

        # Train the model on the training data
        model.fit(X_Train, Y_Train)

        # Make predictions on the test data
        predictions = model.predict(X_Test)

        # Create a DataFrame to compare test and predicted values
        import pandas as pd
        temp = pd.DataFrame(np.vstack((Y_Test, predictions)).transpose())
        temp.columns = ['Test', 'Predicted']

        # Calculate the correctness based on the sign of Test and Predicted values
        temp['Correctness'] = np.where(temp['Test'] * temp['Predicted'] > 0, True, False)

        # Calculate the Winrate as the ratio of correct predictions to the total number of predictions
        Winrate = temp['Correctness'].sum() / temp.shape[0]
        print(Winrate)

    else:
        Winrate = 0

    return Winrate, temp

# Call the function and store the result in 'result'
result = MLPRegressorWinrateCalculation(DataForNN)[1]
