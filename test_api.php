<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Disease API Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-section {
            border: 2px dashed #ddd;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .upload-section:hover {
            border-color: #007bff;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            width: 100%;
            max-width: 400px;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            display: none;
        }
        .success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
        .debug-info {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .confidence-bar {
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }
        .low-confidence {
            background: linear-gradient(90deg, #ffc107, #fd7e14);
        }
        .high-confidence {
            background: linear-gradient(90deg, #28a745, #20c997);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌾 Rice Disease API Test</h1>
        
        <div class="upload-section">
            <h3>Upload Image to Test</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="imageFile" name="file" accept="image/*" required>
                <br>
                <button type="submit" id="submitBtn">🔍 Analyze Image</button>
                <button type="button" id="clearBtn">🗑️ Clear</button>
            </form>
        </div>

        <div id="loading" class="loading" style="display: none;">
            <p>🔄 Analyzing image... Please wait...</p>
        </div>

        <div id="result" class="result"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('imageFile');
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            if (!fileInput.files[0]) {
                alert('Please select an image file');
                return;
            }
            
            // Show loading
            submitBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                // First, try the local API
                let response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    // If local fails, try the remote API
                    response = await fetch('https://riceapi-4n6n.onrender.com/predict', {
                        method: 'POST',
                        body: formData
                    });
                }
                
                const data = await response.json();
                
                // Display result
                displayResult(data);
                
            } catch (error) {
                console.error('Error:', error);
                displayError('Failed to connect to API. Please check if the API is running.');
            } finally {
                submitBtn.disabled = false;
                loading.style.display = 'none';
            }
        });
        
        document.getElementById('clearBtn').addEventListener('click', function() {
            document.getElementById('imageFile').value = '';
            document.getElementById('result').style.display = 'none';
            document.getElementById('loading').style.display = 'none';
        });
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            let html = '';
            
            if (data.status === 'success') {
                const confidence = parseFloat(data.confidence.replace('%', ''));
                const confidenceClass = confidence < 70 ? 'low-confidence' : 'high-confidence';
                
                html = `
                    <div class="result success">
                        <h3>✅ Analysis Complete</h3>
                        <p><strong>Disease:</strong> ${data.disease}</p>
                        <p><strong>Confidence:</strong> ${data.confidence}</p>
                        
                        <div class="confidence-bar">
                            <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
                        </div>
                        
                        ${confidence < 70 ? '<p class="warning">⚠️ Low confidence prediction. Consider retaking the image.</p>' : ''}
                        
                        ${data.debug_info ? `
                            <div class="debug-info">
                                <strong>Debug Information:</strong>
                                ${data.debug_info}
                            </div>
                        ` : ''}
                    </div>
                `;
            } else if (data.status === 'not a rice leaf') {
                html = `
                    <div class="result warning">
                        <h3>🌿 No Rice Leaf Detected</h3>
                        <p>The uploaded image does not appear to contain a rice leaf.</p>
                        <p>Please upload an image of a rice leaf for disease analysis.</p>
                    </div>
                `;
            } else {
                html = `
                    <div class="result error">
                        <h3>❌ Error</h3>
                        <p>${data.error || 'Unknown error occurred'}</p>
                    </div>
                `;
            }
            
            resultDiv.innerHTML = html;
            resultDiv.style.display = 'block';
        }
        
        function displayError(message) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <div class="result error">
                    <h3>❌ Error</h3>
                    <p>${message}</p>
                </div>
            `;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html> 