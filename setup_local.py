#!/usr/bin/env python3
"""
Local Setup Script for Rice Disease API
This script helps you set up and run the API locally for testing.
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip"):
        print("❌ pip not found. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("❌ Failed to install dependencies")
        return False
    
    return True

def check_model_file():
    """Check if the model file exists"""
    model_file = "rice_leaf_model.pth"
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        print(f"✅ Model file found: {model_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_file}")
        print("   Please make sure the model file is in the current directory")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting API server...")
    print("   The server will be available at: http://localhost:8000")
    print("   API docs will be at: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop the server")
    print("\n" + "="*50)
    
    try:
        # Start the server
        subprocess.run([sys.executable, "api.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

def main():
    """Main setup function"""
    print("🌾 Rice Disease API - Local Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check model file
    if not check_model_file():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    print("\n✅ Setup complete! Starting API server...")
    time.sleep(2)
    
    # Start the server
    start_api_server()

if __name__ == "__main__":
    main() 