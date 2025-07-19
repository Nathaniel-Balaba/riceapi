import pytest
from fastapi.testclient import TestClient
from api import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def create_test_image():
    """Create a simple test image"""
    # Create a 64x64 green image (simulating a rice leaf)
    img_array = np.zeros((64, 64, 3), dtype=np.uint8)
    img_array[:, :, 1] = 128  # Green channel
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Rice Leaf Disease Classification API"}

def test_predict_with_valid_image():
    """Test prediction with a valid image"""
    # Create test image
    img_bytes = create_test_image()
    
    # Make request
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "disease" in data
    assert "confidence" in data
    assert "status" in data
    
    # Check if it's a valid response
    assert data["status"] in ["success", "not a rice leaf"]

def test_predict_without_file():
    """Test prediction without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error

def test_predict_with_invalid_file():
    """Test prediction with invalid file type"""
    # Create a text file instead of image
    text_content = b"This is not an image"
    
    response = client.post(
        "/predict",
        files={"file": ("test.txt", text_content, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not a rice leaf"

def test_rice_leaf_detection():
    """Test the rice leaf detection function"""
    from api import is_rice_leaf
    
    # Create a green image (should be detected as rice leaf)
    green_img = Image.new('RGB', (100, 100), color='green')
    assert is_rice_leaf(green_img) == True
    
    # Create a red image (should not be detected as rice leaf)
    red_img = Image.new('RGB', (100, 100), color='red')
    assert is_rice_leaf(red_img) == False

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 