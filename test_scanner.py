import requests
import os
from PIL import Image
import io

def test_api_with_image(image_path):
    """Test the API with a specific image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Prepare the image
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        
        try:
            # Make request to local API (if running locally)
            response = requests.post('http://localhost:8000/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n=== Results for {image_path} ===")
                print(f"Status: {result.get('status')}")
                if result.get('status') == 'success':
                    print(f"Disease: {result.get('disease')}")
                    print(f"Confidence: {result.get('confidence')}")
                elif result.get('status') == 'not a rice leaf':
                    print("No rice leaf detected")
                else:
                    print(f"Error: {result.get('error')}")
            else:
                print(f"HTTP Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to API. Make sure it's running on localhost:8000")
        except Exception as e:
            print(f"Error: {e}")

def test_with_sample_images():
    """Test with different types of images"""
    print("Testing Rice Leaf Disease Scanner API")
    print("=" * 50)
    
    # Test with different images if they exist
    test_images = [
        "bacterial_blight_sample.jpg",
        "brown_spot_sample.jpg", 
        "leaf_smut_sample.jpg",
        "non_leaf_sample.jpg"
    ]
    
    for image in test_images:
        if os.path.exists(image):
            test_api_with_image(image)
        else:
            print(f"\nSkipping {image} - file not found")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_with_sample_images() 