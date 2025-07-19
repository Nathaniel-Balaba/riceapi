# 🌾 Rice Leaf Disease Classification API

A FastAPI-based machine learning API that classifies rice leaf diseases using a CNN model. The system can detect three common rice diseases: Bacterial Leaf Blight, Brown Spot, and Leaf Smut.

## 🚀 Features

- **Disease Classification**: Identifies 3 common rice leaf diseases
- **Image Analysis**: Advanced bias correction and image characteristic analysis
- **Web Interface**: User-friendly PHP frontend with camera and gallery upload
- **API Documentation**: Auto-generated Swagger/OpenAPI docs
- **Local Testing**: Complete setup for local development and testing
- **Bias Correction**: Smart override system to handle model bias

## 🎯 Supported Diseases

1. **Bacterial Leaf Blight** - Yellow/white lesions on leaves
2. **Brown Spot** - Brown/yellow spots with dark borders
3. **Leaf Smut** - Black spots and lesions

## 📋 Prerequisites

- Python 3.8 or higher
- XAMPP (for PHP web interface)
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rice-leaf-diseases.git
cd rice-leaf-diseases
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model File
Make sure you have the `rice_leaf_model.pth` file in the project root directory.

## 🚀 Quick Start

### Option 1: Windows Batch File (Easiest)
```bash
# Double-click start_local.bat
# Or run from command line:
start_local.bat
```

### Option 2: Python Setup Script
```bash
python setup_local.py
```

### Option 3: Manual Start
```bash
python api.py
```

The API will be available at: `http://localhost:8000`

## 🌐 Usage

### Web Interface
1. Start your XAMPP server
2. Open: `http://localhost/rice_leaf_diseases/test_api.php`
3. Upload or capture images
4. View results with detailed debug information

### API Endpoints

#### Predict Disease
```bash
POST /predict
Content-Type: multipart/form-data

# Upload an image file
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Response Format
```json
{
  "disease": "Brown spot",
  "confidence": "78.50%",
  "status": "success",
  "debug_info": "Raw probabilities:\n  Bacterial leaf blight: 15.20%\n  Brown spot: 78.50%\n  Leaf smut: 6.30%\n..."
}
```

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing

### Command Line Testing
```bash
# Test with sample images
python test_local.py

# Test specific image
python test_local.py brownspot_orig_011.jpg
python test_local.py BACTERAILBLIGHT3_009.jpg
```

### Quick Test
```bash
python quick_test.py your_image.jpg
```

## 📁 Project Structure

```
rice-leaf-diseases/
├── api.py                 # FastAPI backend
├── rice_leaf_model.pth    # Trained CNN model
├── requirements.txt       # Python dependencies
├── setup_local.py        # Local setup script
├── start_local.bat       # Windows startup script
├── test_api.php          # PHP web interface
├── test_local.py         # Local testing script
├── quick_test.py         # Quick API test
├── index.php             # Main web interface
├── scan.php              # PHP API proxy
├── train.py              # Model training script
├── LOCAL_TESTING.md      # Local testing guide
└── README.md            # This file
```

## 🔧 Configuration

### Model Parameters
- **Confidence Threshold**: 60% (minimum confidence for prediction)
- **Bias Correction**: Applied to handle model bias towards Leaf Smut
- **Image Analysis**: HSV color space analysis for disease detection

### Rice Leaf Detection
- **Leaf Ratio Threshold**: 0.7
- **Local Variance Threshold**: 250
- **Mean Brightness Threshold**: 70

## 🐛 Troubleshooting

### Common Issues

**❌ "Python not found"**
- Install Python 3.8+ from [python.org](https://python.org)

**❌ "Model file not found"**
- Ensure `rice_leaf_model.pth` is in the project root

**❌ "Port 8000 already in use"**
- Stop other servers or change port in `api.py`

**❌ "Import errors"**
- Run: `pip install -r requirements.txt`

### Debug Information
The API provides detailed debug information including:
- Raw model probabilities
- Bias-corrected probabilities
- Image analysis results
- Override decisions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Rice disease dataset providers
- FastAPI community
- PyTorch team

## 📞 Support

If you encounter any issues or have questions:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [local testing guide](LOCAL_TESTING.md)
3. Open an issue on GitHub

---

**Made with ❤️ for rice farmers and researchers** 