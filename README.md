# 🌾 Rice Leaf Disease Classification API

A FastAPI-based machine learning API that classifies rice leaf diseases using a CNN model. The system can detect three common rice diseases: Bacterial Leaf Blight, Brown Spot, and Leaf Smut.

## 🚀 Live API

**API Endpoint**: https://riceapi-4n6n.onrender.com

**API Documentation**: https://riceapi-4n6n.onrender.com/docs

## 🎯 Supported Diseases

1. **Bacterial Leaf Blight** - Yellow/white lesions on leaves
2. **Brown Spot** - Brown/yellow spots with dark borders  
3. **Leaf Smut** - Black spots and lesions

## 🌐 API Usage

### Predict Disease
```bash
POST /predict
Content-Type: multipart/form-data

# Upload an image file
curl -X POST "https://riceapi-4n6n.onrender.com/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Response Format
```json
{
  "disease": "Brown spot",
  "confidence": "78.50%",
  "status": "success",
  "debug_info": "Raw probabilities:\n  Bacterial leaf blight: 15.20%\n  Brown spot: 78.50%\n  Leaf smut: 6.30%\n..."
}
```

### Example Response
```json
{
  "disease": "Brown spot",
  "confidence": "78.50%",
  "status": "success",
  "debug_info": "Raw probabilities:\n  Bacterial leaf blight: 15.20%\n  Brown spot: 78.50%\n  Leaf smut: 6.30%\n\nCorrected probabilities:\n  Bacterial leaf blight: 76.00%\n  Brown spot: 78.50%\n  Leaf smut: 0.32%\n\nImage analysis suggests: Brown spot\n⚠️ Model bias override applied: Leaf smut → Brown spot"
}
```

## 🔧 Features

- **Disease Classification**: Identifies 3 common rice leaf diseases
- **Bias Correction**: Smart override system to handle model bias
- **Image Analysis**: HSV color space analysis for disease detection
- **Debug Information**: Detailed probability breakdowns
- **Rice Leaf Detection**: Validates input images are rice leaves

## 🛠️ Technical Details

### Model Architecture
- **CNN Model**: Custom PyTorch CNN with 3 convolutional layers
- **Input Size**: 64x64 RGB images
- **Classes**: 3 (Bacterial leaf blight, Brown spot, Leaf smut)

### Bias Correction
The API includes advanced bias correction to handle model bias:
- **Bias Correction Factors**: Applied to balance predictions
- **Image Analysis**: HSV color analysis for disease characteristics
- **Override System**: Corrects model when it's clearly wrong

### Rice Leaf Detection
Validates images contain rice leaves using:
- **Hue Analysis**: Checks for green leaf colors
- **Texture Analysis**: Local variance calculation
- **Brightness Check**: Mean brightness threshold

## 📋 Prerequisites

- Python 3.8 or higher
- PyTorch
- FastAPI
- OpenCV
- PIL (Pillow)

## 🚀 Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rice-leaf-diseases.git
cd rice-leaf-diseases
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
python api.py
```

The API will be available at: `http://localhost:8000`

## 📁 Project Structure

```
rice-leaf-diseases/
├── api.py                 # FastAPI backend
├── rice_leaf_model.pth    # Trained CNN model
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── Procfile              # Render startup command
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

**❌ "502 Bad Gateway"**
- Check if the API is running on Render
- Verify the model file is properly uploaded

**❌ "Model file not found"**
- Ensure `rice_leaf_model.pth` is in the project root

**❌ "Import errors"**
- Check `requirements.txt` has all dependencies

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
2. Open an issue on GitHub

---

**Made with ❤️ for rice farmers and researchers** 