import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from model import CariesDetectionNet, GradCAM, get_transforms

# Page configuration
st.set_page_config(
    page_title="Dental Caries Detection",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .caries-positive {
        background-color: #ffe6e6;
        border-left-color: #ff4444;
    }
    .caries-negative {
        background-color: #e6f7e6;
        border-left-color: #44ff44;
    }
    .confidence-high {
        color: #2E86AB;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F18F01;
        font-weight: bold;
    }
    .confidence-low {
        color: #C73E1D;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained caries detection model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CariesDetectionNet(num_classes=2, pretrained=False, enable_localization=True)
        
        # Try to load the trained model
        if os.path.exists('best_caries_model.pth'):
            model.load_state_dict(torch.load('best_caries_model.pth', map_location=device))
            model.eval()
            st.success("✅ Trained model loaded successfully!")
        else:
            st.warning("⚠️ Trained model not found. Using untrained model for demonstration.")
        
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    _, val_transform = get_transforms()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    input_tensor = val_transform(image).unsqueeze(0)
    return input_tensor, np.array(image)

def predict_caries(model, input_tensor, device):
    """Make prediction on preprocessed image"""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs['classification'], dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Get attention map if available
        attention_map = None
        if 'attention' in outputs:
            attention_map = outputs['attention'][0, 0].cpu().numpy()
    
    return predicted_class, confidence, attention_map, probabilities

def create_visualization(original_image, attention_map, predicted_class, confidence):
    """Create visualization plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Radiograph', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Attention map
    if attention_map is not None:
        im1 = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title('AI Attention Map\n(Regions of Interest)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image, alpha=0.7)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.4)
        axes[2].set_title(f'Overlay Visualization\nPrediction: {"Caries Detected" if predicted_class == 1 else "No Caries"}\nConfidence: {confidence:.1%}', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Attention map\nnot available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(original_image)
        axes[2].set_title(f'Prediction: {"Caries Detected" if predicted_class == 1 else "No Caries"}\nConfidence: {confidence:.1%}', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🦷 Dental Caries Detection System</h1>', unsafe_allow_html=True)
    st.markdown("Upload a dental radiograph to detect the presence of caries using AI")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 Model Information</h2>', unsafe_allow_html=True)
        
        # Load model
        model, device = load_model()
        
        if model is not None:
            st.markdown(f"**Device:** {device}")
            st.markdown("**Model:** ResNet50-based CNN")
            st.markdown("**Features:** Classification + Localization")
            
            st.markdown('<h3 class="sub-header">🎯 How it works</h3>', unsafe_allow_html=True)
            st.markdown("""
            1. **Upload** a dental radiograph
            2. **AI Analysis** using deep learning
            3. **Classification** (Caries/No Caries)
            4. **Attention Map** shows focus areas
            5. **Confidence Score** indicates certainty
            """)
            
            st.markdown('<h3 class="sub-header">📊 Interpretation Guide</h3>', unsafe_allow_html=True)
            st.markdown("""
            - **Red areas** in attention map indicate potential caries
            - **High confidence** (>80%): Very reliable
            - **Medium confidence** (60-80%): Good reliability
            - **Low confidence** (<60%): Requires expert review
            """)
        else:
            st.error("Model failed to load. Please check the model file.")
            return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📁 Upload Radiograph</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a dental radiograph image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear dental radiograph for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Radiograph", use_column_width=True)
            
            # Add image information
            st.markdown(f"**Image Details:**")
            st.markdown(f"- Size: {image.size[0]} × {image.size[1]} pixels")
            st.markdown(f"- Mode: {image.mode}")
            st.markdown(f"- File size: {len(uploaded_file.getvalue())/1024:.1f} KB")
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">🔍 AI Analysis Results</h2>', unsafe_allow_html=True)
            
            # Process image and make prediction
            with st.spinner("🤖 Analyzing radiograph..."):
                input_tensor, original_image = preprocess_image(image)
                predicted_class, confidence, attention_map, probabilities = predict_caries(model, input_tensor, device)
            
            # Display prediction results
            prediction_text = "Caries Detected" if predicted_class == 1 else "No Caries Detected"
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            box_class = "caries-positive" if predicted_class == 1 else "caries-negative"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3>🎯 Prediction: {prediction_text}</h3>
                <p class="confidence-{confidence_level}">Confidence: {confidence:.1%}</p>
                <p><strong>Caries Probability:</strong> {probabilities[0][1].item():.1%}</p>
                <p><strong>No Caries Probability:</strong> {probabilities[0][0].item():.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            if predicted_class == 1:
                if confidence > 0.8:
                    st.error("🚨 **High Risk**: Strong indication of caries. Immediate dental consultation recommended.")
                elif confidence > 0.6:
                    st.warning("⚠️ **Medium Risk**: Possible caries detected. Dental examination advised.")
                else:
                    st.info("ℹ️ **Low Confidence**: Unclear indication. Professional evaluation needed.")
            else:
                if confidence > 0.8:
                    st.success("✅ **Low Risk**: No clear signs of caries detected.")
                else:
                    st.info("ℹ️ **Uncertain**: Results unclear. Regular dental checkup recommended.")
    
    # Visualization section
    if uploaded_file is not None:
        st.markdown('<h2 class="sub-header">📊 Detailed Visualization</h2>', unsafe_allow_html=True)
        
        # Create and display visualization
        fig = create_visualization(original_image, attention_map, predicted_class, confidence)
        st.pyplot(fig)
        
        # Additional analysis tabs
        tab1, tab2, tab3 = st.tabs(["🔍 Detailed Analysis", "📈 Confidence Metrics", "💡 Recommendations"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Analysis Summary:**")
                st.markdown(f"- **Primary Prediction:** {prediction_text}")
                st.markdown(f"- **Confidence Level:** {confidence_level.title()}")
                st.markdown(f"- **Processing Time:** < 1 second")
                
            with col2:
                st.markdown("**Technical Details:**")
                st.markdown(f"- **Model Architecture:** ResNet50 + Custom Heads")
                st.markdown(f"- **Input Resolution:** 224×224 pixels")
                st.markdown(f"- **Attention Map Available:** {'Yes' if attention_map is not None else 'No'}")
        
        with tab2:
            # Confidence visualization
            fig_conf, ax = plt.subplots(1, 1, figsize=(8, 4))
            classes = ['No Caries', 'Caries']
            probs = [probabilities[0][0].item(), probabilities[0][1].item()]
            colors = ['#44ff44', '#ff4444']
            
            bars = ax.bar(classes, probs, color=colors, alpha=0.7)
            ax.set_ylabel('Probability')
            ax.set_title('Class Probabilities')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_conf)
        
        with tab3:
            st.markdown("**Clinical Recommendations:**")
            
            if predicted_class == 1:
                st.markdown("""
                - 🏥 **Schedule immediate dental consultation**
                - 🔍 **Request professional X-ray analysis**
                - 🦷 **Consider preventive treatment options**
                - 📅 **Follow up within 1-2 weeks**
                """)
            else:
                st.markdown("""
                - ✅ **Continue regular oral hygiene routine**
                - 📅 **Maintain scheduled dental checkups**
                - 🦷 **Monitor for any changes or symptoms**
                - 🍭 **Limit sugary foods and drinks**
                """)
            
            st.markdown("**Important Notice:**")
            st.warning("⚠️ This AI system is designed to assist dental professionals and should not replace professional medical diagnosis. Always consult with a qualified dentist for proper evaluation and treatment.")

    # Sample images section
    if uploaded_file is None:
        st.markdown('<h2 class="sub-header">📂 Try Sample Images</h2>', unsafe_allow_html=True)
        st.markdown("No image uploaded yet. You can try the system with sample images from your dataset:")
        
        # List available sample images
        sample_dirs = ['val_caries', 'val_without_caries']
        for dir_name in sample_dirs:
            if os.path.exists(dir_name):
                files = [f for f in os.listdir(dir_name) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    st.markdown(f"**{dir_name.replace('_', ' ').title()}:**")
                    for file in files[:3]:  # Show first 3 files
                        st.markdown(f"- {file}")

if __name__ == "__main__":
    main()
