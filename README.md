# Dental Caries Detection System

A web-based application for detecting dental caries in radiographic images using deep learning.

## Features

🦷 **AI-Powered Detection**: Uses a ResNet50-based CNN for accurate caries classification
🎯 **Attention Visualization**: Shows which areas the AI focuses on
📊 **Confidence Scoring**: Provides reliability metrics for predictions
🖼️ **Easy Upload Interface**: Simple drag-and-drop image upload
📱 **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Option 1: Using Batch File (Windows)
Simply double-click `run_app.bat` to start the application.

### Option 2: Using PowerShell
Right-click `run_app.ps1` and select "Run with PowerShell".

### Option 3: Manual Start
1. Open Command Prompt or PowerShell
2. Navigate to the project directory
3. Run: `C:\Users\Well\Desktop\VSCODE\Carry_Detection\env\Scripts\python.exe -m streamlit run app.py`

## Usage Instructions

1. **Start the Application**: Use one of the methods above
2. **Open Browser**: Navigate to `http://localhost:8501`
3. **Upload Image**: Click "Browse files" and select a dental radiograph
4. **View Results**: See AI prediction, confidence score, and attention map
5. **Interpret Results**: Use the provided guidance for clinical recommendations

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

## Model Information

- **Architecture**: ResNet50 backbone with custom classification and localization heads
- **Input Size**: 224×224 pixels
- **Classes**: Binary classification (Caries/No Caries)
- **Features**: Classification + Attention mapping

## Interface Sections

### 1. Upload Panel
- File upload interface
- Image preview
- File information display

### 2. Results Panel
- AI prediction with confidence score
- Risk assessment
- Probability breakdown

### 3. Visualization
- Original radiograph
- AI attention map (heat map)
- Overlay visualization

### 4. Analysis Tabs
- **Detailed Analysis**: Complete breakdown of results
- **Confidence Metrics**: Visual probability charts
- **Recommendations**: Clinical guidance based on results

## Interpretation Guide

### Confidence Levels
- **High (>80%)**: Very reliable prediction
- **Medium (60-80%)**: Good reliability, consider expert review
- **Low (<60%)**: Requires professional dental evaluation

### Attention Maps
- **Red/Hot areas**: Regions of interest for potential caries
- **Intensity**: Darker red indicates higher AI attention
- **Coverage**: Widespread attention may indicate multiple concerns

## Important Disclaimers

⚠️ **Medical Disclaimer**: This system is designed to assist dental professionals and should not replace professional medical diagnosis. Always consult with a qualified dentist for proper evaluation and treatment.

⚠️ **AI Limitations**: The model's performance depends on image quality, lighting conditions, and similarity to training data.

## Technical Requirements

- Python 3.9+
- PyTorch
- Streamlit
- PIL/Pillow
- OpenCV
- NumPy
- Matplotlib

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `best_caries_model.pth` exists in the project directory
2. **Import errors**: Verify all dependencies are installed
3. **Image upload fails**: Check file format and size
4. **Slow performance**: Consider using GPU acceleration if available

### Performance Tips

- Use high-quality, clear radiographic images
- Ensure proper contrast and brightness
- Crop images to focus on areas of interest
- Use standard radiographic views when possible

## Support

For technical issues or questions about the application, please refer to the model implementation in `model.py` or consult the documentation.

## License

This project is for educational and research purposes. Please ensure compliance with medical device regulations before clinical use.
