const tf = require('@tensorflow/tfjs-node');

// Your data
const data = [
    [1, 12000],
    [8, 11800],
    [15, 11600],
    [22, 11400],
    [29, 11200],
    [5, 11000],
    [12, 10800],
    [19, 10600],
    [26, 10400]
];

// Extracting features (Date) and labels (Invest)
const features = data.map(entry => entry[0]);
const labels = data.map(entry => entry[1]);

// Normalizing the data
const normalize = (arr) => {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return arr.map(value => (value - min) / (max - min));
};

const normalizedFeatures = normalize(features);
const normalizedLabels = normalize(labels);

// Creating a sequential model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Compiling the model
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError', metrics: ['mse'] });

// Converting data to tensors
const xs = tf.tensor2d(normalizedFeatures, [normalizedFeatures.length, 1]);
const ys = tf.tensor2d(normalizedLabels, [normalizedLabels.length, 1]);

// Training the model
model.fit(xs, ys, {
    epochs: 100,
    validationSplit: 0.2,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
            console.log(`Epoch ${epoch + 1}/${100}, MSE: ${logs.mse}, Val MSE: ${logs.val_mse}`);
        }
    }
}).then((history) => {
    // Extracting validation metrics
    const valMSE = history.history.val_mse;

    // Making predictions
    const prediction = model.predict(xs);
    const predictionsArray = Array.from(prediction.dataSync());

    console.log('Validation MSE:', valMSE);

    // Print predictions
    console.log('Predictions:', predictionsArray);
});
