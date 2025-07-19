from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import uvicorn
import cv2

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

# Adjustable thresholds for rice leaf detection
LEAF_RATIO_THRESHOLD = 0.7
LOCAL_VAR_THRESHOLD = 250
MEAN_BRIGHTNESS_THRESHOLD = 70

def is_rice_leaf(image: Image.Image) -> bool:
    try:
        img_array = np.array(image)
        if img_array.shape[0] < 20 or img_array.shape[1] < 20:
            print("Rejected: Image too small")
            return False
            
        small_img = image.resize((50, 50))
        hsv_img = small_img.convert('HSV')
        hsv_array = np.array(hsv_img)
        hue_channel = hsv_array[:,:,0].flatten()
        saturation_channel = hsv_array[:,:,1].flatten()
        valid_hue_points = hue_channel[saturation_channel > 50]
        
        if len(valid_hue_points) > 0:
            leaf_hue_points = ((valid_hue_points >= 20) & (valid_hue_points <= 140)).sum()
            leaf_ratio = leaf_hue_points / len(valid_hue_points)
            
            gray_img = image.convert('L')
            gray_array = np.array(gray_img)
            local_var = np.var(gray_array)
            mean_brightness = np.mean(gray_array)

            print(f"Leaf ratio: {leaf_ratio:.2f}, Local var: {local_var:.2f}, Mean brightness: {mean_brightness:.2f}")

            if (leaf_ratio > LEAF_RATIO_THRESHOLD and
                local_var > LOCAL_VAR_THRESHOLD and
                mean_brightness > MEAN_BRIGHTNESS_THRESHOLD):
                print("Accepted as rice leaf")
                return True
            else:
                print("Rejected: Did not meet thresholds")
        else:
            print("Rejected: No valid hue points")
        
        return False
    except Exception as e:
        print(f"Error in is_rice_leaf: {e}")
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

# More balanced bias correction factors - not too aggressive
BIAS_CORRECTION = {
    0: 2.0,   # Boost Bacterial leaf blight moderately
    1: 2.5,   # Boost Brown spot moderately  
    2: 0.3    # Reduce Leaf smut but not too much
}

def analyze_image_characteristics(image):
    """Analyze image characteristics to help determine the likely disease"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Calculate standard deviations
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Analyze for disease characteristics
        characteristics = {
            'avg_hue': h_mean,
            'avg_saturation': s_mean,
            'avg_value': v_mean,
            'hue_variation': h_std,
            'saturation_variation': s_std,
            'value_variation': v_std
        }
        
        print(f"Image characteristics: {characteristics}")
        return characteristics
        
    except Exception as e:
        print(f"Error analyzing image characteristics: {e}")
        return None

def get_likely_disease_from_characteristics(characteristics):
    """Estimate likely disease based on image characteristics"""
    if not characteristics:
        return None
    
    h_mean = characteristics['avg_hue']
    s_mean = characteristics['avg_saturation']
    v_mean = characteristics['avg_value']
    h_std = characteristics['hue_variation']
    
    # More refined heuristics based on typical disease appearances
    # These thresholds are adjusted based on actual disease characteristics
    
    # Brown spot typically has brown/yellow colors (lower hue, moderate saturation, moderate value)
    if h_mean < 25 and s_mean > 60 and v_mean < 140 and v_mean > 80:
        return "Brown spot"
    
    # Bacterial leaf blight typically has yellow/white lesions (higher hue, lower saturation, higher value)
    elif h_mean > 35 and s_mean < 70 and v_mean > 130:
        return "Bacterial leaf blight"
    
    # Leaf smut typically has black spots (very low value, low saturation)
    elif v_mean < 70 and s_mean < 50:
        return "Leaf smut"
    
    # If none of the above, return None (let model decide)
    return None

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
            
            # Analyze image characteristics
            characteristics = analyze_image_characteristics(image)
            likely_disease = get_likely_disease_from_characteristics(characteristics)
            
            # Get raw probabilities for all classes for debugging
            raw_probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            corrected_probs = probabilities[0]
            
            # Get the model's prediction
            _, predicted = torch.max(probabilities, 1)
            model_prediction = class_names[predicted.item()]
            model_confidence = probabilities[0][predicted.item()].item() * 100
            
            print(f"\nModel prediction: {model_prediction} with confidence: {model_confidence:.2f}%")
            print(f"Likely disease from image analysis: {likely_disease}")
            
            # Check if model is clearly wrong (predicting Leaf smut when image analysis suggests otherwise)
            # Only override if there's a strong disagreement and image analysis is confident
            if (model_prediction == "Leaf smut" and 
                likely_disease and 
                likely_disease != "Leaf smut" and
                model_confidence > 80):  # Only override if model is very confident about Leaf smut
                
                print(f"⚠️ Model bias detected! Overriding prediction from '{model_prediction}' to '{likely_disease}'")
                
                # Override the prediction
                if likely_disease == "Brown spot":
                    predicted = torch.tensor([1])  # Brown spot index
                elif likely_disease == "Bacterial leaf blight":
                    predicted = torch.tensor([0])  # Bacterial leaf blight index
                
                # Adjust confidence based on image analysis
                confidence = min(75.0, model_confidence * 0.8)  # Reduce confidence when overriding
                print(f"Overridden prediction: {class_names[predicted.item()]} with confidence: {confidence:.2f}%")
            else:
                # Use model prediction but cap confidence
                confidence = min(model_confidence, 85.0)
                print(f"Using model prediction: {model_prediction} with confidence: {confidence:.2f}%")
            
            # Create debug info string
            debug_info = f"Raw probabilities:\n"
            for i, class_name in enumerate(class_names):
                raw_prob = raw_probs[i].item() * 100
                debug_info += f"  {class_name}: {raw_prob:.2f}%\n"
            
            debug_info += f"\nCorrected probabilities:\n"
            for i, class_name in enumerate(class_names):
                corrected_prob = corrected_probs[i].item() * 100
                debug_info += f"  {class_name}: {corrected_prob:.2f}%\n"
            
            if likely_disease:
                debug_info += f"\nImage analysis suggests: {likely_disease}\n"
                if model_prediction == "Leaf smut" and likely_disease != "Leaf smut":
                    debug_info += f"⚠️ Model bias override applied: {model_prediction} → {class_names[predicted.item()]}\n"
            
            # Print debugging info about the probabilities
            print(f"\nRaw probabilities: {[f'{raw_probs[i].item()*100:.2f}%' for i in range(len(class_names))]}") 
            print(f"Corrected probabilities: {[f'{corrected_probs[i].item()*100:.2f}%' for i in range(len(class_names))]}") 
            print(f"Final prediction: {class_names[predicted.item()]} with confidence: {confidence:.2f}%\n")
            
            # Add detailed breakdown for debugging
            print("Detailed breakdown:")
            for i, class_name in enumerate(class_names):
                raw_prob = raw_probs[i].item() * 100
                corrected_prob = corrected_probs[i].item() * 100
                print(f"  {class_name}: Raw={raw_prob:.2f}%, Corrected={corrected_prob:.2f}%")
            print()
            
            # Additional check: if the highest probability is still too dominant, adjust
            max_prob = corrected_probs.max().item()
            if max_prob > 0.9:  # Only correct if probability is >90% (was 80%)
                print(f"Warning: Very high confidence detected ({max_prob*100:.2f}%). Applying additional correction.")
                # Reduce the highest probability and redistribute
                second_highest_idx = (corrected_probs == corrected_probs.topk(2)[0][1]).nonzero().item()
                corrected_probs[predicted.item()] *= 0.8  # Reduce highest by 20% (was 30%)
                corrected_probs[second_highest_idx] *= 1.3  # Boost second highest by 30% (was 50%)
                # Renormalize
                corrected_probs = corrected_probs / corrected_probs.sum()
                # Recalculate prediction
                _, predicted = torch.max(corrected_probs, 0)
                confidence = corrected_probs[predicted.item()].item() * 100
                confidence = min(confidence, 85.0)  # Cap again
                print(f"After additional correction: {class_names[predicted.item()]} with confidence: {confidence:.2f}%")
                
                # Update debug info with final corrected probabilities
                debug_info += f"\nFinal corrected probabilities (after high-confidence adjustment):\n"
                for i, class_name in enumerate(class_names):
                    final_prob = corrected_probs[i].item() * 100
                    debug_info += f"  {class_name}: {final_prob:.2f}%\n"
            
            # Check if confidence is above threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = class_names[predicted.item()]
            else:
                prediction = "Unknown"
                confidence = 0.0  # Set confidence to 0 for unknown cases
            
            return {
                "disease": prediction,
                "confidence": f"{confidence:.2f}%",
                "status": "success",
                "debug_info": debug_info
            }

    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 