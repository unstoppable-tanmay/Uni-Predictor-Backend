import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('data1.csv')

# Split the data into features (X) and target variable (y)
X = data[['date', 'invest', 'customer','companyValuation']]
y = data['profitLoss']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(X_test).flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')

# Plot the original values vs. predicted values
plt.scatter(X_test['date'], y_test, label='Original Values')
plt.scatter(X_test['date'], y_pred, label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Profit Loss')
plt.legend()
plt.title('Original Values vs. Predicted Values')
plt.show()
