const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const cors = require('cors');

// Initialize express app
const app = express();

const PORT = 3002;
let image_path;
let pythonProcess = null; 

// Enable CORS for the frontend (running on 127.0.0.1:3000)
app.use(cors({
    origin: ['http://localhost:3000', 'http://127.0.0.1:3000']  // Allow both origins
}));
// Configure multer for file uploads, storing images with original extensions
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname); // Get file extension
        const filename = `${file.fieldname}-${Date.now()}${ext}`;
        cb(null, filename);
    },
});

const upload = multer({ storage });

// Serve frontend files from 'public' folder and output folder
app.use(express.static('public/obj_detection_app'));
app.use('/output', express.static(path.join(__dirname, 'output')));  // Serve the output folder

app.use(express.json());


app.use(cors({
    origin: ['http://localhost:3000'],  // Allow React frontend
}));

// Route to stream webcam processed by YOLOv5
app.get('/webcam', (req, res) => {
    const action = req.query.action; // Read action query parameter

    if (action === 'start') {
        // If a Python process is already running, prevent multiple processes
        if (pythonProcess) {
            return res.status(400).json({ error: 'Webcam detection is already running.' });
        }

        const yolov5Script = path.resolve('./yolov5/detect.py');
        const outputFolder = './output';

        // Ensure the output directory exists
        if (!fs.existsSync(outputFolder)) {
            fs.mkdirSync(outputFolder, { recursive: true });
        }

        // Start the YOLOv5 process
        pythonProcess = spawn('python3', [
            yolov5Script,
            '--weights', './yolov5/yolov5s.pt', // YOLOv5 weights file
            '--img', '640',                    // Image size
            '--conf', '0.4',                  // Confidence threshold
            '--source', '0',                   // Use webcam as source
            '--project', outputFolder,         // Output folder
            '--name', 'webcam_results',        // Output folder name
        ]);

        // Set up the HTTP response as a multipart MJPEG stream
        res.setHeader('Content-Type', 'multipart/x-mixed-replace; boundary=frame');

        pythonProcess.stdout.on('data', (data) => {
            // Each frame sent by YOLOv5 script is output as MJPEG frame
            const imageBuffer = Buffer.from(data);
            res.write(`--frame\r\n`);
            res.write(`Content-Type: image/jpeg\r\n\r\n`);
            res.write(imageBuffer);
            res.write(`\r\n`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python process finished with exit code ${code}`);
            pythonProcess = null; // Reset process tracker
        });

        pythonProcess.on('error', (err) => {
            console.error('Python process error:', err);
            res.status(500).json({ error: 'Failed to start Python process', details: err.message });
        });

        console.log('Webcam detection started...');
    } else if (action === 'stop') {
        // Stop the Python process
        if (pythonProcess) {
            pythonProcess.kill('SIGTERM'); // Send termination signal
            pythonProcess = null;
            console.log('Webcam detection stopped.');
            res.status(200).json({ message: 'Webcam detection stopped.' });
        } else {
            res.status(400).json({ error: 'No webcam detection process is running.' });
        }
    } else {
        res.status(400).json({ error: 'Invalid action. Use ?action=start or ?action=stop' });
    }
});


// Route to handle object detection
app.post('/detect', upload.single('image'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }

    const imagePath = path.resolve(req.file.path); // Get the correct file path
    const yolov5Script = path.resolve('./yolov5/detect.py'); // Path to the YOLOv5 detect.py script

    // Generate a unique identifier for the output folder using a hash
    const uniqueId = crypto.randomBytes(16).toString('hex');
    const outputFolder = path.resolve('./output', uniqueId); // Unique folder for this request

    // Ensure the output directory exists
    if (!fs.existsSync(outputFolder)) {
        fs.mkdirSync(outputFolder, { recursive: true });
    }

    // Call YOLOv5's detect.py script
    const pythonProcess = spawn('python3.11', [
        yolov5Script,
        '--weights', './yolov5/yolov5s.pt', // Path to the YOLOv5 weights file
        '--img', '640',  // Image size
        '--conf', '0.4', // Confidence threshold
        '--source', imagePath,  // Input image path
        '--project', outputFolder, // Output directory (unique for each request)
        '--name', 'results', // Output folder name
    ]);

    let errorMessage = '';
    let resultMessage = '';

    // Capture data from stdout (regular output from Python)
    pythonProcess.stdout.on('data', (data) => {
        resultMessage += data.toString();
        console.log(data.toString()); // Log Python output
    });

    // Capture data from stderr (error output from Python)
    pythonProcess.stderr.on('data', (data) => {
        errorMessage += data.toString();
        console.error(data.toString()); // Log Python error output
    });

    // Handle process close event
    pythonProcess.on('close', (code) => {
        // Cleanup the uploaded image file after processing
        fs.unlink(imagePath, (err) => {
            if (err) console.error('Failed to delete uploaded file:', err);
        });

        console.log(`Python process finished with exit code ${code}`);

        if (code !== 0) {
            console.error('Python process error:', errorMessage);
            return res.status(500).json({ error: 'Detection failed', details: errorMessage });
        }

        // Path to the 'results' folder where YOLOv5 saves the annotated images
        const resultFolder = path.join(outputFolder, 'results');

        // Get all images in the result folder
        fs.readdir(resultFolder, (err, files) => {
            if (err) {
                console.error('Error reading result folder:', err);
                return res.status(500).json({ error: 'Failed to read result folder' });
            }

            // Filter out non-image files (optional)
            const imageFiles = files.filter(file => ['.jpg', '.jpeg', '.png', '.bmp'].includes(path.extname(file).toLowerCase()));

            if (imageFiles.length === 0) {
                console.error('No images found in result folder');
                return res.status(500).json({ error: 'No images generated' });
            }

            // Generate image URLs for the client to access
            const imageUrls = imageFiles.map(file => `http://localhost:3002/output/${uniqueId}/results/${file}`);

            console.log('Generated image URLs:', imageUrls);
            image_path = imageUrls;

            // Send a list of image URLs to the frontend
            res.send({ status: 200, image: image_path });
        });
    });

    // Timeout to avoid hanging processes
    pythonProcess.on('error', (err) => {
        console.error('Python process error:', err);
        res.status(500).json({ error: 'Failed to start the Python process', details: err.message });
    });

    // Log Python process start
    console.log('Python process started for object detection...');
});


// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
