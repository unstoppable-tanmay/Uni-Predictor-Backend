const express = require('express');
const multer = require('multer');
const csvParser = require('csv-parser');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const tf = require('@tensorflow/tfjs-node');
const jwt = require('jsonwebtoken');
const fs = require('fs').promises;
var bodyParser = require('body-parser')
const cookieParser = require('cookie-parser');

const app = express();
app.use(bodyParser.urlencoded({ extended: false })) // parse application/x-www-form-urlencoded
app.use(bodyParser.json()) // parse application/json
app.use(cookieParser());
const port = 3000;
const secretKey = 'JWT-CODE'; // Replace with your secret key

mongoose.connect('mongodb://localhost:27017/UniPredictor', { useNewUrlParser: true, useUnifiedTopology: true });
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
db.once('open', () => console.log('Connected to MongoDB'));

const userSchema = new mongoose.Schema({
    username: String,
    email: String,
    password: String, // Hashed password
    dataFilePath: String,
    modelStatus: String, // 'training', 'trained', etc.
    modelMaxMin: {
        xsMin: [Number], // Define as an array of numbers
        xsMax: [Number],
        ysMin: [Number],
        ysMax: [Number],
    }
});

const User = mongoose.model('User', userSchema);

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Middleware for user authentication
const authenticateUser = (req, res, next) => {
    const token = req.cookies.token;

    if (!token) {
        return res.status(401).send('Access denied. No token provided.');
    }

    jwt.verify(token, secretKey, async (err, user) => {
        if (err) {
            return res.status(403).send('Invalid token.');
        }

        req.user = await User.findById(user.id)

        next();
    });
};

app.post('/register', async (req, res) => {
    console.log(req.body);

    // User Must Send Required Details
    if (!req.body.username || !req.body.email || !req.body.password) {
        return res.status(404).send('Details Required');
    }

    const { username, email, password } = req.body;

    try {
        const hashedPassword = await bcrypt.hash(password, 10);

        const newUser = new User({
            username,
            email,
            password: hashedPassword,
            dataFilePath: null,
            modelStatus: '', // Set the initial status
        });

        await newUser.save();

        // Generate a JWT token for the new user
        const token = jwt.sign({ username: newUser.username }, secretKey);

        // Set the token as a cookie
        res.cookie('token', token, { httpOnly: true });

        return res.status(201).send('User registered successfully');
    } catch (error) {
        console.error(error);
        return res.status(500).send('Internal Server Error');
    }
});

app.post('/login', async (req, res) => {
    const { username, password } = req.body;

    const user = await User.findOne({ username });

    if (!user || !await bcrypt.compare(password, user.password)) {
        return res.status(401).send('Invalid credentials');
    }

    const token = jwt.sign({ username: user.username, id: user._id }, secretKey);

    // Set the token as a cookie
    res.cookie('token', token, { httpOnly: true });

    res.status(200).json({ token });
});

app.post('/upload', authenticateUser, upload.single('file'), async (req, res) => {
    try {
        const data = req.file.buffer.toString('utf-8');
        const parsedData = [];

        // ... (same parsing logic as before)
        const lines = data.split('\n');
        for (const line of lines) {
            const [date, invest, profitLoss, customer, availabilityShortage, companyValuation, workCapacity] = line.split(',');
            parsedData.push({
                date: Number(date.split(' ')[1]),
                invest: parseFloat(invest),
                profitLoss: parseFloat(profitLoss),
                customer: customer,
                availabilityShortage: availabilityShortage,
                companyValuation: companyValuation !== 'null' ? parseFloat(companyValuation) : null,
                workCapacity: parseFloat(workCapacity),
            });
        }

        const user = req.user;

        // Update user dataFilePath and modelStatus
        const fileName = `${user.username}_data.csv`;
        const filePath = `./files/${fileName}`;
        await fs.writeFile(filePath, data);

        user.modelStatus = 'training';

        await user.save();

        // Start model training in the background
        trainModel(user, parsedData);

        res.status(200).send('File uploaded. Model is building in the background.');
    } catch (error) {
        console.error(error);
        res.status(502).send('Internal Server Error');
    }
});

// Function to train the model
async function trainModel(user, parsedData) {
    // Function to normalize the data
    function normalizeData(tensor) {
        const min = tensor.min(0);
        const max = tensor.max(0);
        const range = max.sub(min);

        // Conditionally handle zero range to avoid division by zero and NaN
        const normalized = tf.where(
            range.equal(0),
            tf.fill(tensor.shape, 0),  // Fallback to fill with zeros when range is zero
            tensor.sub(min).div(range)
        );

        return { normalized, min, max };
    }

    try {
        // Fetch all data
        const data = parsedData.slice(1, parsedData.length);

        // Extract features (X) and labels (Y)
        const xs = tf.tensor2d(data.map(item => [item.date]));
        const ys = tf.tensor2d(data.map(item => [item.invest, item.profitLoss, Number(item.customer), item.availabilityShortage == 'true' ? 1 : 0, item.companyValuation, item.workCapacity]));

        // Create and compile the model
        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [1], units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 6, activation: 'linear' }));

        // Compile the model
        model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

        // Normalize xs and ys
        const { normalized: xsNormalized, min: xsMin, max: xsMax } = normalizeData(xs);
        const { normalized: ysNormalized, min: ysMin, max: ysMax } = normalizeData(ys);

        // Train the model
        await model.fit(xsNormalized, ysNormalized, {
            epochs: 100,
            validationSplit: 0.2,
        });

        // Save the trained model
        const fileName = `${user.username}_model`;
        const modelPath = `./models/${fileName}`;
        await model.save(`file://${modelPath}`);

        // Update user modelStatus to 'trained'
        user.modelStatus = 'trained';
        // Convert tensors to arrays for modelMaxMin
        user.modelMaxMin = {
            xsMin: xsMin.arraySync()[0],
            xsMax: xsMax.arraySync()[0],
            ysMin: ysMin.arraySync(),
            ysMax: ysMax.arraySync(),
        };

        await user.save();
        console.log({
            xsMin: xsMin.arraySync()[0],
            xsMax: xsMax.arraySync()[0],
            ysMin: ysMin.arraySync(),
            ysMax: ysMax.arraySync(),
        })
    } catch (error) {
        console.error(error);
    }
}

app.get('/predict/:weekNumber', authenticateUser, async (req, res) => {
    try {
        const user = req.user;

        // Check the model status
        if (user.modelStatus !== 'trained') {
            return res.status(400).send('Model is not ready yet. Please wait for training to complete.');
        }

        const folderName = `${user.username}_model`;
        const modelPath = `./models/${folderName}/model.json`;

        const weekNumber = parseInt(req.params.weekNumber);
        const model = await tf.loadLayersModel(`file://${modelPath}`); // Load the model
        const input = tf.tensor2d([[weekNumber]]);
        const prediction = model.predict(input);

        console.log(prediction.arraySync())

        // Inverse normalize the predicted values
        const predictionValues = tf.unstack(prediction)[0].dataSync();

        // Assuming user.modelMaxMin contains the min-max values for ys (output) in the format { ysMin, ysMax }
        const { ysMin, ysMax } = user.modelMaxMin;

        // Perform inverse normalization
        const inverseNormalizedValues = predictionValues.map((value, index) => {
            if (index === 3) {
                // Convert availabilityShortage back to boolean
                return value >= 0.5;
            } else {
                // Inverse normalization for other features
                return value * (ysMax[index] - ysMin[index]) + ysMin[index];
            }
        });

        console.log(inverseNormalizedValues)

        const [invest, profitLoss, customer, availabilityShortage, companyValuation, predictedWorkCapacity] = inverseNormalizedValues;

        return res.status(200).json({
            // weekNumber: weekNumber,
            // invest: invest, 
            profitLoss: profitLoss,
            // customer: customer,
            // availabilityShortage: availabilityShortage,
            // companyValuation: companyValuation,
            // predictedWorkCapacity: predictedWorkCapacity
        });
    } catch (error) {
        console.error(error);
        res.status(500).send('Internal Server Error');
    }
});


app.listen(port, () => console.log(`Server is running on port ${port}`));
