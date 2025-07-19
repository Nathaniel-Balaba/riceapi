# 🌾 Rice Disease API - Local Testing Guide

## Quick Start (Windows)

### Option 1: Using Batch File (Easiest)
1. **Double-click** `start_local.bat`
2. **Wait** for dependencies to install
3. **Server starts** automatically at `http://localhost:8000`

### Option 2: Using Python Script
1. **Open Command Prompt** in this folder
2. **Run**: `python setup_local.py`
3. **Server starts** automatically

### Option 3: Manual Setup
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start server**: `python api.py`
3. **Server runs** at `http://localhost:8000`

## Testing the API

### 1. Web Interface
- **Open**: `http://localhost/rice_leaf_diseases/test_api.php`
- **Upload images** and see results with debug info

### 2. API Documentation
- **Open**: `http://localhost:8000/docs`
- **Test directly** in the Swagger UI

### 3. Command Line Testing
```bash
# Test with sample images
python test_local.py

# Test specific image
python test_local.py brownspot_orig_011.jpg
python test_local.py BACTERAILBLIGHT3_009.jpg
```

### 4. Quick Test Script
```bash
python quick_test.py brownspot_orig_011.jpg
```

## What to Expect

### ✅ Fixed Issues:
- **No more 100% confidence** (capped at 85%)
- **Brown spot images** → Should show "Brown spot" (not "Leaf smut")
- **Bacterial blight images** → Should show "Bacterial leaf blight"
- **Debug information** shows bias correction working

### 🔍 Debug Output:
- **Raw probabilities** from model
- **Corrected probabilities** after bias adjustment
- **Image analysis** results
- **Override decisions** when model is wrong

### 🎯 Test Images:
- `brownspot_orig_011.jpg` → Should predict "Brown spot"
- `BACTERAILBLIGHT3_009.jpg` → Should predict "Bacterial leaf blight"
- `leafsmut_orig_001.jpg` → Should predict "Leaf smut"

## Troubleshooting

### ❌ "Python not found"
- Install Python 3.8+ from [python.org](https://python.org)

### ❌ "Model file not found"
- Make sure `rice_leaf_model.pth` is in the same folder

### ❌ "Port 8000 already in use"
- Stop other servers or change port in `api.py`

### ❌ "Import errors"
- Run: `pip install -r requirements.txt`

## Server URLs

- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Test Page**: http://localhost/rice_leaf_diseases/test_api.php

## Stopping the Server

- **Press Ctrl+C** in the terminal
- Or **close the terminal window** 