import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('data1.csv')

# Split the data into features (X) and target variable (y)
X = data[['date', 'invest', 'customer', 'companyValuation']]
y = data['profitLoss']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a more complex model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model for more epochs
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(X_test).flatten()

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Define a custom accuracy function based on a percentage threshold
def custom_accuracy(y_true, y_pred, threshold_percentage=50):
    threshold = threshold_percentage / 100
    upper_limit = y_true * (1 + threshold)
    lower_limit = y_true * (1 - threshold)
    correct_predictions = ((y_pred <= upper_limit) & (y_pred >= lower_limit)).sum()
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

# Calculate custom accuracy
custom_acc = custom_accuracy(y_test, y_pred, threshold_percentage=50)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2) Score: {r2}')
print(f'Custom Accuracy (within 50%): {custom_acc}')

# Plot the original values vs. predicted values
plt.scatter(X_test['date'], y_test, label='Original Values')
plt.scatter(X_test['date'], y_pred, label='Predicted Values')
plt.xlabel('Date')
plt.ylabel('Profit Loss')
plt.legend()
plt.title('Original Values vs. Predicted Values')
plt.show()