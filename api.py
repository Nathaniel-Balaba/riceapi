from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import uvicorn

# Define the CNN model (same structure as what was used to train the model)
class RiceLeafCNN(nn.Module):
    def __init__(self, num_classes):
        super(RiceLeafCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

app = FastAPI(title="Rice Leaf Disease Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model and set it to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RiceLeafCNN(num_classes=3).to(device)  # 3 classes as per your dataset
model.load_state_dict(torch.load('rice_leaf_model.pth', map_location=device))
model.eval()

# Rice leaf detection function
def is_rice_leaf(image: Image.Image) -> bool:
    """
    Function to detect if the image contains a rice leaf.
    Uses color features and texture analysis to make a determination.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        bool: True if the image likely contains a rice leaf, False otherwise
    """
    try:
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Check if image is too small or too large (unusual for leaf images)
        if img_array.shape[0] < 20 or img_array.shape[1] < 20:
            return False
            
        # Extract dominant colors
        # Rice leaves are typically green to yellow-brown depending on disease
        # Resize for faster processing
        small_img = image.resize((50, 50))
        small_array = np.array(small_img)
        
        # Calculate average color in the HSV space (better for color analysis)
        hsv_img = small_img.convert('HSV')
        hsv_array = np.array(hsv_img)
        
        # Check if dominant hue is in green-yellow range (20-140 in HSV)
        # This range covers healthy and diseased rice leaves
        hue_channel = hsv_array[:,:,0].flatten()
        saturation_channel = hsv_array[:,:,1].flatten()
        
        # Filter out pixels with low saturation (not colorful enough)
        valid_hue_points = hue_channel[saturation_channel > 50]
        
        if len(valid_hue_points) > 0:
            # Check if a significant portion of the image has leaf-like colors
            leaf_hue_points = ((valid_hue_points >= 20) & (valid_hue_points <= 140)).sum()
            leaf_ratio = leaf_hue_points / len(valid_hue_points)
            
            # If more than 30% of the valid colored pixels have leaf-like hues
            if leaf_ratio > 0.3:
                # Additional check: texture variance (leaves have texture)
                gray_img = image.convert('L')
                gray_array = np.array(gray_img)
                # Calculate local variance as a simple texture measure
                local_var = np.var(gray_array)
                
                # Leaves typically have some texture variation
                if local_var > 100:
                    return True
        
        return False
    except Exception:
        # If any error occurs during analysis, be conservative and return False
        return False

# Define a more robust image transform with better preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Keep same size for compatibility with existing model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Define confidence threshold and bias correction factors
CONFIDENCE_THRESHOLD = 60.0  # Lowered to allow detection of more diseases

# Bias correction factors to adjust for model bias toward Leaf smut
# These factors help balance the predictions based on observed bias
BIAS_CORRECTION = {
    0: 1.15,  # Boost Bacterial leaf blight predictions
    1: 1.20,  # Boost Brown spot predictions even more
    2: 0.75   # Reduce Leaf smut predictions
}

@app.get("/")
async def root():
    return {"message": "Welcome to Rice Leaf Disease Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Rice leaf detection step
        is_leaf = is_rice_leaf(image)
        if not is_leaf:
            return {
                "disease": "Unknown",
                "confidence": "0.00%",
                "status": "not a rice leaf"
            }

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction with bias correction
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Apply bias correction to outputs before calculating probabilities
            corrected_outputs = outputs.clone()
            for class_idx, correction_factor in BIAS_CORRECTION.items():
                corrected_outputs[0, class_idx] *= correction_factor
            
            probabilities = torch.nn.functional.softmax(corrected_outputs, dim=1)
            
            # Get class with highest probability after bias correction
            _, predicted = torch.max(probabilities, 1)
            confidence = probabilities[0][predicted.item()].item() * 100
            
            # Get raw probabilities for all classes for debugging
            raw_probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            corrected_probs = probabilities[0]
            
            # Print debugging info about the probabilities
            print(f"\nRaw probabilities: {[f'{raw_probs[i].item()*100:.2f}%' for i in range(len(class_names))]}") 
            print(f"Corrected probabilities: {[f'{corrected_probs[i].item()*100:.2f}%' for i in range(len(class_names))]}") 
            print(f"Predicted class: {class_names[predicted.item()]} with confidence: {confidence:.2f}%\n")
            
            # Check if confidence is above threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = class_names[predicted.item()]
            else:
                prediction = "Unknown"
                confidence = 0.0  # Set confidence to 0 for unknown cases

        return {
            "disease": prediction,
            "confidence": f"{confidence:.2f}%",
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 