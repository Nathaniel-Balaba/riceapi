<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Scanner</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }
        .camera-container {
            position: relative;
            width: 100%;
            height: 400px;
            background: #000;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
        }
        #camera-feed, #preview-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .button-group {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            background-color: #3498db;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .hidden {
            display: none;
        }
        #camera-error {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 4px;
            text-align: center;
        }
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rice Leaf Disease Scanner</h1>
        
        <div class="camera-container">
            <video id="camera-feed" autoplay playsinline muted></video>
            <img id="preview-image" class="hidden" alt="Preview">
            <div id="camera-error" class="hidden">
                <p id="error-message">Camera access required</p>
            </div>
        </div>

        <div class="button-group" id="main-controls">
            <button id="camera-btn">Take Photo</button>
            <button id="gallery-btn">Upload Image</button>
        </div>

        <div class="button-group hidden" id="action-buttons">
            <button id="retake-btn">Retake</button>
            <button id="analyze-btn">Analyze</button>
        </div>

        <div id="result-content" class="result hidden"></div>

        <input type="file" id="gallery-upload" accept="image/*" class="hidden">
        <canvas id="photo-canvas" class="hidden"></canvas>
        
        <div id="loading" class="hidden">
            <div>Analyzing image... Please wait...</div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const videoElement = document.getElementById('camera-feed');
            const previewImage = document.getElementById('preview-image');
            const cameraError = document.getElementById('camera-error');
            const errorMessage = document.getElementById('error-message');
            const galleryBtn = document.getElementById('gallery-btn');
            const cameraBtn = document.getElementById('camera-btn');
            const fileInput = document.getElementById('gallery-upload');
            const photoCanvas = document.getElementById('photo-canvas');
            const mainControls = document.getElementById('main-controls');
            const actionButtons = document.getElementById('action-buttons');
            const retakeBtn = document.getElementById('retake-btn');
            const analyzeBtn = document.getElementById('analyze-btn');
            const loadingIndicator = document.getElementById('loading');
            const resultContent = document.getElementById('result-content');
            
            let stream = null;
            let capturedImage = null;

            async function setupCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: false
                    });
                    videoElement.srcObject = stream;
                    cameraError.classList.add('hidden');
                } catch (err) {
                    console.error('Error accessing camera:', err);
                    cameraError.classList.remove('hidden');
                    errorMessage.textContent = err.name === 'NotAllowedError' 
                        ? 'Camera access denied. Please allow camera access to use this feature.'
                        : 'Could not access camera. Please make sure your device has a working camera.';
                }
            }

            setupCamera();

            galleryBtn.addEventListener('click', () => fileInput.click());

            fileInput.addEventListener('change', function(e) {
                if (e.target.files && e.target.files[0]) {
                    const file = e.target.files[0];
                    if (!file.type.startsWith('image/')) {
                        alert('Please select an image file (JPG, PNG, etc).');
                        fileInput.value = '';
                        return;
                    }
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        videoElement.classList.add('hidden');
                        previewImage.src = e.target.result;
                        previewImage.classList.remove('hidden');
                        cameraError.classList.add('hidden');
                        capturedImage = file;
                        mainControls.classList.add('hidden');
                        actionButtons.classList.remove('hidden');
                    };
                    reader.readAsDataURL(file);
                }
            });

            cameraBtn.addEventListener('click', function() {
                const context = photoCanvas.getContext('2d');
                photoCanvas.width = videoElement.videoWidth;
                photoCanvas.height = videoElement.videoHeight;
                context.drawImage(videoElement, 0, 0, photoCanvas.width, photoCanvas.height);
                
                photoCanvas.toBlob(function(blob) {
                    capturedImage = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' });
                    previewImage.src = photoCanvas.toDataURL('image/jpeg');
                    videoElement.classList.add('hidden');
                    previewImage.classList.remove('hidden');
                    mainControls.classList.add('hidden');
                    actionButtons.classList.remove('hidden');
                }, 'image/jpeg');
            });

            retakeBtn.addEventListener('click', function() {
                videoElement.classList.remove('hidden');
                previewImage.classList.add('hidden');
                mainControls.classList.remove('hidden');
                actionButtons.classList.add('hidden');
                resultContent.classList.add('hidden');
                capturedImage = null;
            });

            analyzeBtn.addEventListener('click', async function() {
                if (!capturedImage) return;

                loadingIndicator.classList.remove('hidden');
                resultContent.classList.add('hidden');

                const formData = new FormData();
                formData.append('image', capturedImage);

                try {
                    const response = await fetch('scan.php', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    console.log('API Response:', result); // Debug log
                    
                    resultContent.innerHTML = '';
                    
                    if (result.status === 'error') {
                        resultContent.classList.add('error');
                        resultContent.classList.remove('success');
                        resultContent.textContent = result.error || 'An error occurred';
                    } else if (result.status === 'not a rice leaf') {
                        resultContent.classList.add('error');
                        resultContent.classList.remove('success');
                        resultContent.innerHTML = '<strong>No rice leaf detected</strong><br>Please try again with a clear image of a rice leaf.';
                    } else if (result.status === 'success') {
                        resultContent.classList.add('success');
                        resultContent.classList.remove('error');
                        
                        const confidence = parseFloat(result.confidence);
                        let resultText = `<strong>Disease:</strong> ${result.disease}<br><strong>Confidence:</strong> ${result.confidence}`;
                        
                        if (confidence < 80) {
                            resultText += '<br><br><em>Warning: Low confidence prediction. Please ensure you have a clear, well-lit image of a rice leaf.</em>';
                        }
                        
                        resultContent.innerHTML = resultText;
                    } else {
                        // Handle unexpected response format
                        resultContent.classList.add('error');
                        resultContent.classList.remove('success');
                        resultContent.textContent = 'Unexpected response from server. Please try again.';
                    }
                } catch (error) {
                    console.error('Error:', error); // Debug log
                    resultContent.classList.add('error');
                    resultContent.classList.remove('success');
                    resultContent.textContent = 'Error analyzing image. Please try again.';
                } finally {
                    loadingIndicator.classList.add('hidden');
                    resultContent.classList.remove('hidden');
                }
            });
        });
    </script>
</body>
</html> 