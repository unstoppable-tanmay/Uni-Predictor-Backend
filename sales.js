const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const csv = require('csv-parser');
const nodeplotlib = require('nodeplotlib');

// Set seed for reproducibility
tf.setBackend('tensorflow');

// CustomScaler class
class CustomScaler {
    constructor(min, max) {
        this.min = min;
        this.max = max;
    }

    apply(data) {
        return tf.div(tf.sub(data, this.min), tf.sub(this.max, this.min));
    }

    inverseTransform(scaledData) {
        return tf.add(tf.mul(scaledData, tf.sub(this.max, this.min)), this.min);
    }

    // Method to get scalar data for saving
    getScalerData() {
        return { min: this.min.arraySync(), max: this.max.arraySync() };
    }
}

// Load and preprocess the dataset using csv-parser
async function loadDataset() {
    const rows = [];
    return new Promise((resolve, reject) => {
        fs.createReadStream('airline-passengers.csv')
            .pipe(csv())
            .on('data', (row) => rows.push(row))
            .on('end', () => {
                const values = rows.map(item => parseFloat(item.Passengers));
                const dataset = tf.tensor2d([values]).transpose();

                // Calculate min and max for custom scaler
                const min = tf.min(dataset);
                const max = tf.max(dataset);

                // Create custom scaler
                const scaler = new CustomScaler(min, max);

                // Apply custom scaler
                const normalizedData = scaler.apply(dataset);

                resolve({ dataset: normalizedData, scaler });
            })
            .on('error', (error) => reject(error));
    });
}

// Create the LSTM model
function createModel(inputShape) {
    const model = tf.sequential();

    model.add(tf.layers.lstm({ units: 16, inputShape }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({ loss: 'meanSquaredError', optimizer: 'adam' });

    return model;
}

// Function to save the model
async function saveModel(model, modelPath) {
    const saveResult = await model.save(`file://${modelPath}`);
    console.log('Model saved:', saveResult);
}

// Function to load the model
async function loadModel(modelPath) {
    const loadedModel = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully');
    return loadedModel;
}

// Function to make predictions using the loaded model
function predict(model, inputData, scaler) {
    // Normalize the input data using the scaler
    const normalizedInput = scaler.apply(tf.tensor2d([inputData]));

    // Reshape the input tensor to be 3D [batch_size, timesteps, features]
    const inputTensor = normalizedInput.reshape([1, normalizedInput.shape[0], 1]);

    // Make predictions using the loaded model
    const prediction = model.predict(inputTensor);
    const predictionData = prediction.arraySync();

    // Invert predictions using the scaler
    const invertedPrediction = scaler.inverseTransform(tf.tensor2d(predictionData, [predictionData.length, 1])).arraySync();

    console.log('Inverted Prediction:', invertedPrediction);
}

// Train the model
async function trainModel(model, trainX, trainY, epochs = 10, batchSize = 1) {
    await model.fit(trainX, trainY, {
        epochs,
        batchSize,
        verbose: 1,
    });
}

// Function to make predictions and evaluate accuracy with graph
function evaluateModel(model, testX, testY, scaler, lookBack, dataset, trainX, trainY) {
    const testPredict = model.predict(testX).arraySync();
    const trainPredict = model.predict(trainX).arraySync()
    const testYArray = testY.arraySync();
    const trainYArray = trainY.arraySync();

    // Invert predictions
    const invertedTestPredict = scaler
        .inverseTransform(tf.tensor2d(testPredict, [testPredict.length, 1]))
        .arraySync();

    // Invert true values
    const invertedTestY = scaler
        .inverseTransform(tf.tensor2d(testYArray, [testYArray.length, 1]))
        .arraySync();

    // Invert predictions
    const invertedTrainPredict = scaler
        .inverseTransform(tf.tensor2d(trainPredict, [trainPredict.length, 1]))
        .arraySync();

    // Invert true values
    const invertedTrainY = scaler
        .inverseTransform(tf.tensor2d(trainYArray, [trainYArray.length, 1]))
        .arraySync();

    // Calculate custom accuracy
    const accuracy = customAccuracy(invertedTestY, invertedTestPredict);
    console.log(`Test Accuracy: ${accuracy.toFixed(2)}`);

    // console.log(invertedTestPredict, invertedTestY, invertedTrainPredict, invertedTrainY)

    return { invertedTestPredict, invertedTestY };
}

// Custom accuracy function
function customAccuracy(yTrue, yPred) {
    const percentageDiff = yTrue.map((trueVal, index) => {
        return Math.abs((trueVal - yPred[index]) / trueVal) * 100;
    });

    const accuracy =
        percentageDiff.filter(diff => diff <= 15).length / percentageDiff.length;

    return accuracy;
}

// Function to create datasets
function createDatasets(dataset, lookBack) {
    const dataX = [];
    const dataY = [];

    for (let i = 0; i < dataset.length - lookBack - 1; i++) {
        const a = dataset.slice(i, i + lookBack);
        const b = dataset[i + lookBack];

        dataX.push(a);
        dataY.push(b);
    }

    const splitIndex = Math.floor(dataset.length * 0.8);

    const trainX = tf.tensor3d(dataX.slice(0, splitIndex), [splitIndex, lookBack, 1]);
    const trainY = tf.tensor2d(dataY.slice(0, splitIndex), [splitIndex, 1]);

    const testX = tf.tensor3d(dataX.slice(splitIndex), [dataset.length - splitIndex - lookBack - 1, lookBack, 1]);
    const testY = tf.tensor2d(dataY.slice(splitIndex), [dataset.length - splitIndex - lookBack - 1, 1]);

    return [trainX, trainY, testX, testY];
}

// Save scaler as JSON
function saveScaler(scaler, scalerPath) {
    const scalerData = scaler.getScalerData();
    const scalerJson = JSON.stringify(scalerData);

    fs.writeFileSync(scalerPath, scalerJson);
    console.log('Scaler saved:', scalerPath);
}

// Load scaler from JSON
function loadScaler(scalerPath) {
    const scalerJson = fs.readFileSync(scalerPath, 'utf-8');
    const scalerData = JSON.parse(scalerJson);

    // Create a CustomScaler instance based on the loaded data
    const scaler = new CustomScaler(tf.scalar(scalerData.min), tf.scalar(scalerData.max));

    console.log('Scaler loaded successfully');
    return scaler;
}

// Main execution
async function main() {
    const { dataset, scaler } = await loadDataset();

    const lookBack = 1;
    const [trainX, trainY, testX, testY] = createDatasets(dataset.arraySync(), lookBack);

    const model = createModel([lookBack, 1]);
    await trainModel(model, trainX, trainY);

    // Save the trained model
    const modelPath = "models/model"
    await saveModel(model, modelPath);

    // Save scaler
    const scalerPath = 'models/model/scaler.json';
    saveScaler(scaler, scalerPath);

    evaluateModel(model, testX, testY, scaler, lookBack, dataset, trainX, trainY);
}

// 
async function predictMain() {
    const modelPath = "./models/model/model.json"
    const scalerPath = "./models/model/scaler.json"

    // Load the saved model
    const loadedModel = await loadModel(modelPath);
    const loadedScaler = loadScaler(scalerPath);

    // Example input data for prediction
    const inputData = [355];

    // Make predictions using the loaded model
    predict(loadedModel, inputData, loadedScaler);
}

main();
// predictMain()