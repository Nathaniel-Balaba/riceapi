import requests
import sys

def test_api(image_path):
    """Quick test of the API with detailed output"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/jpeg')}
            response = requests.post('http://localhost:8000/predict', files=files)
            
            print(f"\n{'='*50}")
            print(f"Testing: {image_path}")
            print(f"{'='*50}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success!")
                print(f"Status: {result.get('status')}")
                if result.get('status') == 'success':
                    print(f"🎯 Disease: {result.get('disease')}")
                    print(f"📊 Confidence: {result.get('confidence')}")
                elif result.get('status') == 'not a rice leaf':
                    print("🌿 No rice leaf detected")
                else:
                    print(f"❌ Error: {result.get('error')}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"❌ File not found: {image_path}")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        print("   Run: python api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_api(image_path)
    else:
        print("Usage: python quick_test.py <image_path>")
        print("Example: python quick_test.py brownspot_orig_011.jpg") 