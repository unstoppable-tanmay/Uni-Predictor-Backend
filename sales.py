import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# # fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset
dataframe = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print(trainX,trainY)


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)


# calculate accuracy
accuracy = model.evaluate(testX, testY, verbose=0)
print('Test Accuracy: %.2f' % (accuracy))


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


def custom_accuracy(y_true, y_pred):
    # calculate percentage difference between predicted and original data
	percentage_diff = np.abs((y_true - y_pred) / y_true) * 100
    
    # calculate accuracy as the inverse proportion of samples with percentage difference <= 10%
	accuracy = np.sum(percentage_diff <= 10) / len(y_true)
    
	return accuracy

# calculate custom accuracy
train_accuracy = custom_accuracy(trainY[0], trainPredict[:,0])
test_accuracy = custom_accuracy(testY[0], testPredict[:,0])

print('Train Accuracy: %.2f' % (train_accuracy))
print('Test Accuracy: %.2f' % (test_accuracy))


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label='Baseline')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.legend()
plt.show()


# Function to predict with the loaded scaler
def predict_with_scaler(scaler, input_data):
    # Normalize input data with the loaded scaler
    normalized_data = scaler.transform(input_data)

    # Reshape normalized data
    normalized_data = np.reshape(normalized_data, (normalized_data.shape[0], 1, normalized_data.shape[1]))

    # Make predictions with the original model
    predictions = model.predict(normalized_data)

    # Invert predictions with the loaded scaler
    inverted_predictions = scaler.inverse_transform(predictions)

    return inverted_predictions

new_data = np.array([[231]])  # Replace with your own data
predicted_values = predict_with_scaler(scaler, new_data)
print("Predicted Values:", predicted_values)