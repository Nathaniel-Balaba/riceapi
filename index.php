<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Scanner</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            touch-action: manipulation;
            font-family: Arial, sans-serif;
        }
        .hidden {
            display: none;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
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
        .floating-modal {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            width: 400px;
            max-height: 80vh;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            z-index: 50;
        }
    </style>
</head>
<body class="bg-gray-100">
  <main class="relative min-h-screen w-full overflow-hidden flex flex-col pb-28">
    <!-- Camera View -->
    <div class="relative flex-grow bg-black" id="camera-container">
      <video id="camera-feed" autoplay playsinline muted class="h-full w-full object-cover"></video>
      
      <!-- Camera permission error message -->
      <div id="camera-error" class="absolute inset-0 flex items-center justify-center bg-black text-white p-4 text-center hidden">
        <p id="error-message">Camera access required</p>
      </div>
      
      <!-- Scanning overlay -->
      <div class="absolute inset-0 pointer-events-none">
        <div class="h-full w-full border-2 border-white/20">
          <!-- Optional: Add scanning animation or guides here -->
        </div>
      </div>

      <!-- Preview image (when captured) -->
      <img id="preview-image" class="h-full w-full object-contain hidden" alt="Preview">
    </div>

    <!-- Back button -->
    <div class="absolute top-4 left-4 z-10">
      <button id="back-btn" class="rounded-full bg-black/30 backdrop-blur-sm text-white hover:bg-black/40 p-3">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
          <path d="m12 19-7-7 7-7"></path>
          <path d="M19 12H5"></path>
        </svg>
      </button>
    </div>

    <!-- Bottom controls -->
    <div class="fixed bottom-4 left-0 right-0 flex justify-center gap-6 z-20">
      <!-- Gallery button -->
      <button id="gallery-btn" class="rounded-full h-14 w-14 bg-green-500 text-white hover:bg-green-600 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
          <rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect>
          <circle cx="9" cy="9" r="2"></circle>
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
        </svg>
        <span class="sr-only">Upload from gallery</span>
      </button>

      <!-- Camera button -->
      <button id="camera-btn" class="rounded-full h-16 w-16 bg-green-500 text-white hover:bg-green-600 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-8 w-8">
          <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"></path>
          <circle cx="12" cy="13" r="3"></circle>
        </svg>
        <span class="sr-only">Take photo</span>
      </button>

      <!-- Scan button -->
      <button id="scan-btn" class="rounded-full h-14 w-14 bg-green-500 text-white hover:bg-green-600 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
          <path d="M3 7V5a2 2 0 0 1 2-2h2"></path>
          <path d="M17 3h2a2 2 0 0 1 2 2v2"></path>
          <path d="M21 17v2a2 2 0 0 1-2 2h-2"></path>
          <path d="M7 21H5a2 2 0 0 1-2-2v-2"></path>
          <rect width="10" height="10" x="7" y="7" rx="2"></rect>
        </svg>
        <span class="sr-only">Scan</span>
      </button>
    </div>

    <!-- Action buttons (shown after capture) -->
    <div id="action-buttons" class="fixed bottom-4 left-0 right-0 flex justify-center gap-6 z-20 hidden">
      <button id="retake-btn" class="rounded-full h-14 w-14 bg-green-500 text-white hover:bg-green-600 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
          <path d="M3 2v6h6"></path>
          <path d="M3 13a9 9 0 1 0 3-7.7L3 8"></path>
        </svg>
        <span class="sr-only">Retake</span>
      </button>

      <button id="analyze-btn" class="rounded-full h-16 w-16 bg-blue-500 text-white hover:bg-blue-600 flex items-center justify-center">
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-8 w-8">
          <path d="m9 11 3 3L22 4"></path>
          <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path>
        </svg>
        <span class="sr-only">Analyze</span>
      </button>
    </div>

    <!-- Hidden file input for gallery upload -->
    <input type="file" id="gallery-upload" accept="image/*" class="hidden">

    <!-- Canvas for capturing photos (hidden) -->
    <canvas id="photo-canvas" class="hidden"></canvas>

    <!-- Loading indicator -->
    <div id="loading" class="absolute inset-0 bg-black/70 flex items-center justify-center z-20 hidden">
      <div class="text-white text-center">
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-white"></div>
        <p class="mt-2">Scanning image... Please wait...</p>
      </div>
    </div>

    <!-- Results panel (floating modal) -->
    <div id="results-panel" class="floating-modal bg-white flex flex-col hidden overflow-hidden">
      <div class="p-4 bg-gray-100 flex items-center">
        <button id="close-results" class="p-2 rounded-full hover:bg-gray-200">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-6 w-6">
            <path d="m12 19-7-7 7-7"></path>
            <path d="M19 12H5"></path>
          </svg>
        </button>
        <h2 class="text-lg font-bold ml-2">Scan Results</h2>
      </div>
      <div class="flex-grow overflow-auto p-4 max-h-[60vh]">
        <div id="result-content" class="result"></div>
      </div>
    </div>
    
    <!-- Modal backdrop -->
    <div id="modal-backdrop" class="absolute inset-0 bg-black/50 z-40 hidden"></div>
  </main>

  <script>
    document.addEventListener('DOMContentLoaded', function() {
      // Elements
      const videoElement = document.getElementById('camera-feed');
      const previewImage = document.getElementById('preview-image');
      const cameraError = document.getElementById('camera-error');
      const errorMessage = document.getElementById('error-message');
      const galleryBtn = document.getElementById('gallery-btn');
      const cameraBtn = document.getElementById('camera-btn');
      const scanBtn = document.getElementById('scan-btn');
      const backBtn = document.getElementById('back-btn');
      const fileInput = document.getElementById('gallery-upload');
      const photoCanvas = document.getElementById('photo-canvas');
      const actionButtons = document.getElementById('action-buttons');
      const retakeBtn = document.getElementById('retake-btn');
      const analyzeBtn = document.getElementById('analyze-btn');
      const loadingIndicator = document.getElementById('loading');
      const resultsPanel = document.getElementById('results-panel');
      const closeResults = document.getElementById('close-results');
      const resultContent = document.getElementById('result-content');
      const modalBackdrop = document.getElementById('modal-backdrop');
      
      let stream = null;
      let capturedImage = null;

      // Initialize camera
      setupCamera();

      // Setup camera
      async function setupCamera() {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: 'environment', // Use back camera if available
              width: { ideal: 1920 },
              height: { ideal: 1080 }
            },
            audio: false
          });

          videoElement.srcObject = stream;
          cameraError.classList.add('hidden');
        } catch (err) {
          console.error('Error accessing camera:', err);
          cameraError.classList.remove('hidden');
          
          if (err.name === 'NotAllowedError') {
            errorMessage.textContent = 'Camera access denied. Please allow camera access to use this feature.';
          } else {
            errorMessage.textContent = 'Could not access camera. Please make sure your device has a working camera.';
          }
        }
      }

      // Gallery button click handler
      galleryBtn.addEventListener('click', function() {
        fileInput.click();
      });

      // File input change handler
      fileInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          const reader = new FileReader();
          
          reader.onload = function(e) {
            // Hide video, show preview
            videoElement.classList.add('hidden');
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            
            // Store the file for later use
            capturedImage = file;
            
            // Show action buttons, hide camera controls
            document.querySelector('.fixed.bottom-4').classList.add('hidden');
            actionButtons.classList.remove('hidden');
          };
          
          reader.readAsDataURL(file);
        }
      });

      // Camera button click handler
      cameraBtn.addEventListener('click', function() {
        if (!stream) return;
        
        // Get video dimensions
        const width = videoElement.videoWidth;
        const height = videoElement.videoHeight;
        
        // Set canvas dimensions to match video
        photoCanvas.width = width;
        photoCanvas.height = height;
        
        // Draw current video frame to canvas
        const context = photoCanvas.getContext('2d');
        context.drawImage(videoElement, 0, 0, width, height);
        
        // Convert canvas to image data URL
        const imageDataUrl = photoCanvas.toDataURL('image/jpeg');
        
        // Hide video, show preview
        videoElement.classList.add('hidden');
        previewImage.src = imageDataUrl;
        previewImage.classList.remove('hidden');
        
        // Store the image data for later use
        photoCanvas.toBlob((blob) => {
          capturedImage = blob;
        }, 'image/jpeg', 0.95);
        
        // Show action buttons, hide camera controls
        document.querySelector('.fixed.bottom-4').classList.add('hidden');
        actionButtons.classList.remove('hidden');
      });

      // Retake button click handler
      retakeBtn.addEventListener('click', function() {
        // Hide preview, show video
        previewImage.classList.add('hidden');
        videoElement.classList.remove('hidden');
        
        // Show camera controls, hide action buttons
        document.querySelector('.fixed.bottom-4').classList.remove('hidden');
        actionButtons.classList.add('hidden');
        
        // Clear captured image
        capturedImage = null;
      });

      // Analyze button click handler
      analyzeBtn.addEventListener('click', function() {
        if (!capturedImage) return;
        
        const formData = new FormData();
        formData.append('image', capturedImage, 'capture.jpg');
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        
        // Send to backend for analysis
        scanImage(formData);
      });

      // Scan button click handler
      scanBtn.addEventListener('click', function() {
        // This button initiates a live scan without capturing
        // For simplicity, we'll just capture and analyze in one step
        if (!stream) return;
        
        // Get video dimensions
        const width = videoElement.videoWidth;
        const height = videoElement.videoHeight;
        
        // Set canvas dimensions to match video
        photoCanvas.width = width;
        photoCanvas.height = height;
        
        // Draw current video frame to canvas
        const context = photoCanvas.getContext('2d');
        context.drawImage(videoElement, 0, 0, width, height);
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        
        // Convert to blob and analyze
        photoCanvas.toBlob((blob) => {
          const formData = new FormData();
          formData.append('image', blob, 'capture.jpg');
          scanImage(formData);
        }, 'image/jpeg', 0.95);
      });

      // Back button click handler
      backBtn.addEventListener('click', function() {
        // If results are showing, close them
        if (!resultsPanel.classList.contains('hidden')) {
          resultsPanel.classList.add('hidden');
          modalBackdrop.classList.add('hidden');
          return;
        }
        
        // If preview is showing, go back to camera
        if (!previewImage.classList.contains('hidden')) {
          retakeBtn.click();
          return;
        }
        
        // Otherwise, go back to previous page
        window.history.back();
      });

      // Close results button click handler
      closeResults.addEventListener('click', function() {
        resultsPanel.classList.add('hidden');
        modalBackdrop.classList.add('hidden');
      });

      // Modal backdrop click handler
      modalBackdrop.addEventListener('click', function() {
        resultsPanel.classList.add('hidden');
        modalBackdrop.classList.add('hidden');
      });

      // Clean up on page unload
      window.addEventListener('beforeunload', function() {
        if (stream) {
          stream.getTracks().forEach(track => track.stop());
        }
      });

      // Function to scan image
      function scanImage(formData) {
        fetch('scan.php', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          // Hide loading indicator
          loadingIndicator.classList.add('hidden');
          
          // Show results panel and backdrop
          resultsPanel.classList.remove('hidden');
          modalBackdrop.classList.remove('hidden');
          
          if (data.status === 'not a rice leaf') {
            resultContent.className = 'result error';
            resultContent.innerHTML = `
              <div class="p-4 bg-yellow-100 text-yellow-800 rounded-md">
                <h3 class="font-bold mb-2">No rice leaf detected</h3>
                <p>Please try again with a clear image of a rice leaf.</p>
              </div>
            `;
            resultContent.style.display = 'block';
            return;
          }
          
          if (data.status === 'success') {
            resultContent.className = 'result success';
            let resultHtml = `
              <div class="mb-4">
                <h3 class="text-xl font-bold mb-2">Disease Detected:</h3>
                <p class="text-2xl font-bold text-green-700">${data.disease}</p>
              </div>
              <div class="mb-4">
                <h4 class="font-bold">Confidence:</h4>
                <p class="text-lg">${data.confidence}</p>
              </div>
            `;
            
            if (data.all_probabilities) {
              resultHtml += '<div class="mt-4"><h4 class="font-bold mb-2">All Probabilities:</h4><ul class="space-y-1">';
              for (const [disease, prob] of Object.entries(data.all_probabilities)) {
                resultHtml += `<li class="flex justify-between"><span>${disease}</span><span>${prob}</span></li>`;
              }
              resultHtml += '</ul></div>';
            }
            
            resultContent.innerHTML = resultHtml;
            resultContent.style.display = 'block';
          } else {
            resultContent.className = 'result error';
            resultContent.innerHTML = `
              <div class="p-4 bg-red-100 text-red-800 rounded-md">
                <h3 class="font-bold mb-2">Error</h3>
                <p>
                  ${data.error || 'An unknown error occurred'}
                </p>
              </div>
            `;
            resultContent.style.display = 'block';
          }
        })
        .catch(error => {
          // Hide loading indicator
          loadingIndicator.classList.add('hidden');
          
          // Show results panel with error
          resultsPanel.classList.remove('hidden');
          modalBackdrop.classList.remove('hidden');
          resultContent.className = 'result error';
          resultContent.innerHTML = `
            <div class="p-4 bg-red-100 text-red-800 rounded-md">
              <h3 class="font-bold mb-2">Error</h3>
              <p>${error.message || 'An unknown error occurred'}</p>
            </div>
          `;
          resultContent.style.display = 'block';
        });
      }
    });
  </script>
</body>
</html> 