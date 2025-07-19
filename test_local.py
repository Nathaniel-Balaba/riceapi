import requests
import os
import sys

def test_local_api(image_path):
    """Test the local API with detailed output"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return
            
        print(f"\n{'='*60}")
        print(f"🧪 Testing: {image_path}")
        print(f"{'='*60}")
        
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/jpeg')}
            response = requests.post('http://localhost:8000/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API Response:")
                print(f"   Status: {result.get('status')}")
                
                if result.get('status') == 'success':
                    print(f"   🎯 Disease: {result.get('disease')}")
                    print(f"   📊 Confidence: {result.get('confidence')}")
                    
                    # Show debug info if available
                    if result.get('debug_info'):
                        print(f"\n🔍 Debug Information:")
                        print(result.get('debug_info'))
                        
                elif result.get('status') == 'not a rice leaf':
                    print("   🌿 No rice leaf detected")
                else:
                    print(f"   ❌ Error: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(response.text)
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to local API")
        print("   Make sure to run: python api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🌾 Rice Disease API Local Test")
    print("=" * 40)
    
    # Test with common image names
    test_images = [
        "brownspot_orig_011.jpg",
        "BACTERAILBLIGHT3_009.jpg", 
        "leafsmut_orig_001.jpg"
    ]
    
    # Check if specific image provided
    if len(sys.argv) > 1:
        test_images = [sys.argv[1]]
    
    for image in test_images:
        test_local_api(image)
        
    print(f"\n{'='*60}")
    print("🎯 Test Summary:")
    print("   - If you see 'Model bias override applied', the fix is working!")
    print("   - Brown spot images should show 'Brown spot' not 'Leaf smut'")
    print("   - Bacterial blight images should show 'Bacterial leaf blight'")
    print("   - Confidence should be capped at 85% maximum")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 