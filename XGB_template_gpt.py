import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download bitcoin price data from Yahoo
df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1483228800&period2=1614863200&interval=1d&events=history')
df.sort_values(by='Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Calculate the simple moving average for the past 30, 60, and 90 days
df['SMA_30'] = df['Close'].rolling(window=30).mean()
df['SMA_60'] = df['Close'].rolling(window=60).mean()
df['SMA_90'] = df['Close'].rolling(window=90).mean()

# Set the future returns as the label
df['Future_Return'] = df['Close'].shift(-1) / df['Close'] - 1
df['Future_Return_Class'] = np.where(df['Future_Return'] > 0, 1, 0)
df.dropna(inplace=True)

# Split the data into feature matrix and label
X = df.drop(columns=['Date', 'Future_Return', 'Close', 'Future_Return_Class'])
y = df['Future_Return_Class']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Define the hyperparameters to be searched over
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300]
}

# Create a GridSearchCV object and fit it to the data
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', verbose=0)
grid_search.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error", eval_set=[(X_val, y_val)])

# Print the best parameters found by the grid search
print('Best parameters:', grid_search.best_params_)

# Make predictions on the validation set
y_pred = grid_search.predict(X_val)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy score:', accuracy)
