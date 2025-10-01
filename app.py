import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from model import VGG16CariesDetectionNet, get_transforms
from segformer_model import SegFormerCariesDetection

# Page configuration
st.set_page_config(
    page_title="Dental Caries Detection",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state for app reset
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'uploaded_file_key' not in st.session_state:
    st.session_state.uploaded_file_key = 0

# Premium Dark Theme with Sophisticated Elegance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .main {
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .hero-section {
        background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
        padding: 3.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(30, 64, 175, 0.3);
        border: 1px solid rgba(30, 64, 175, 0.2);
    }
    
    .hero-title {
        color: #f1f5f9;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }
    
    .hero-description {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 400;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem;
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
        margin: 2rem 0;
        border: 1px solid #334155;
    }
    
    .image-info {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        color: #e2e8f0;
    }
    
    .result-positive {
        background: linear-gradient(135deg, #2d1b3d 0%, #3f1f47 100%);
        border: 3px solid #ef4444;
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 50px rgba(239, 68, 68, 0.25);
        position: relative;
        overflow: hidden;
        color: #fef2f2;
    }
    
    .result-positive::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #ef4444, #dc2626, #b91c1c);
    }
    
    .result-negative {
        background: linear-gradient(135deg, #1a2e1a 0%, #2d4a2d 100%);
        border: 3px solid #22c55e;
        padding: 3rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 50px rgba(34, 197, 94, 0.25);
        position: relative;
        overflow: hidden;
        color: #f0fdf4;
    }
    
    .result-negative::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #22c55e, #16a34a, #15803d);
    }
    
    .confidence-score {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1.2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    .probability-score {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .analysis-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 3rem;
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
        margin: 2rem 0;
        border: 1px solid #334155;
    }
    
    .metrics-container {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 1px solid #475569;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .visualization-container {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        margin: 2rem 0;
        border: 1px solid #4b5563;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 2rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        color: #e2e8f0;
    }
    
    .footer-section {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        margin-top: 3rem;
        border-top: 6px solid #3b82f6;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4);
        color: #cbd5e1;
    }
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.4rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        letter-spacing: 0.02em;
    }
    
    .reset-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        border: 1px solid #334155;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
        margin: 3rem 0;
    }
    
    /* Streamlit component styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        border: 2px dashed #6b7280 !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        border: 2px dashed #6b7280 !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader div[data-testid="stFileUploaderDropzone"] > div {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* Force dark background on all file uploader elements */
    .stFileUploader {
        background: transparent !important;
    }
    
    .stFileUploader > div {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        border: 2px dashed #6b7280 !important;
        border-radius: 16px !important;
    }
    
    .stFileUploader section {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        border: 2px dashed #6b7280 !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader section > div {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* File uploader text and icon styling */
    .stFileUploader small {
        color: #cbd5e1 !important;
    }
    
    .stFileUploader p {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader svg {
        fill: #6b7280 !important;
    }
    
    /* Browse files button styling */
    .stFileUploader button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Hide the top header bar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    .stApp > header {
        display: none !important;
    }
    
    /* Hide the deployment toolbar */
    .stDeployButton {
        display: none !important;
    }
    
    .css-1rs6os {
        display: none !important;
    }
    
    .css-17eq0hr {
        display: none !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    .stMetric {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4b5563;
        color: #e2e8f0 !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #1a2e1a 0%, #2d4a2d 100%) !important;
        border: 1px solid #22c55e !important;
        color: #dcfce7 !important;
    }
    
    .stSuccess > div {
        color: #dcfce7 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #2d1b3d 0%, #3f1f47 100%) !important;
        border: 1px solid #ef4444 !important;
        color: #fef2f2 !important;
    }
    
    .stError > div {
        color: #fef2f2 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #3d2914 0%, #4a3619 100%) !important;
        border: 1px solid #f59e0b !important;
        color: #fef3c7 !important;
    }
    
    .stWarning > div {
        color: #fef3c7 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: #dbeafe !important;
    }
    
    .stInfo > div {
        color: #dbeafe !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    p {
        color: #e2e8f0 !important;
    }
    
    /* Enhanced text visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown div {
        color: #e2e8f0 !important;
    }
    
    .stText, .stCaption {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader label {
        color: #e2e8f0 !important;
    }
    
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #e2e8f0 !important;
    }
    
    /* Metric labels and values */
    .metric-container label {
        color: #cbd5e1 !important;
    }
    
    .metric-container div {
        color: #f1f5f9 !important;
    }
    
    /* Sidebar text */
    .css-1d391kg {
        color: #e2e8f0 !important;
    }
    
    /* General streamlit text */
    div[data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
    
    /* File uploader text */
    .stFileUploader div {
        color: #e2e8f0 !important;
    }
    
    /* Progress text */
    .stProgress div {
        color: #e2e8f0 !important;
    }
    
    /* Spinner text */
    .stSpinner div {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_type="vgg16"):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "segformer":
        # Load SegFormer model from trained checkpoint
        try:
            model = SegFormerCariesDetection(model_path='./results/checkpoint-26955')
            return model, device
        except Exception as e:
            st.error(f"Error loading SegFormer model: {str(e)}")
            # Fall back to VGG16 if SegFormer fails
            model_type = "vgg16"
    
    if model_type == "vgg16":
        # Load VGG16 model (original implementation)
        model = VGG16CariesDetectionNet(num_classes=2, pretrained=False, enable_localization=True)
        if os.path.exists('final_vgg16_caries_model.pth'):
            model.load_state_dict(torch.load('final_vgg16_caries_model.pth', map_location=device))
            model.eval()
        model.to(device)
    
    return model, device

def analyze_image(model, image, device, model_type="vgg16"):
    """Analyze image for caries"""
    # First, check the object type of the model to determine how to handle it
    if model_type == "segformer" and hasattr(model, 'predict'):
        # Use SegFormer model's predict function directly
        result = model.predict(image)
        return (
            result['predicted_class'],
            result['confidence'],
            result['caries_prob'],
            result['attention_map']
        )
    else:
        # Original VGG16 inference or fallback for any model without predict method
        _, val_transform = get_transforms()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs['classification'], dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            attention_map = None
            if 'attention' in outputs:
                attention_map = outputs['attention'][0, 0].cpu().numpy()
        
        return predicted_class, confidence, probabilities[0][1].item(), attention_map

def show_results(image, predicted_class, confidence, caries_prob, attention_map):
    """Display analysis results with elegant styling and organized layout"""
    
    # Result header with enhanced styling
    if predicted_class == 1:
        st.markdown(f"""
        <div class="result-positive">
            <h2>🚨 Caries Detected</h2>
            <div class="confidence-score">{confidence:.0%}</div>
            <div class="probability-score">Caries Probability: {caries_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional recommendations in organized cards
        if confidence > 0.8:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🏥 High Confidence Detection</h4>
                <p><strong>Recommendation:</strong> Schedule dental appointment immediately</p>
                <p><em>Urgent professional evaluation recommended</em></p>
            </div>
            """, unsafe_allow_html=True)
        elif confidence > 0.6:
            st.markdown("""
            <div class="recommendation-card">
                <h4>⚠️ Moderate Confidence Detection</h4>
                <p><strong>Recommendation:</strong> Consider dental examination soon</p>
                <p><em>Professional consultation advised within 1-2 weeks</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🔍 Low Confidence Detection</h4>
                <p><strong>Recommendation:</strong> Monitor and consider professional evaluation</p>
                <p><em>Regular dental checkups recommended</em></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <h2>✅ No Caries Detected</h2>
            <div class="confidence-score">{confidence:.0%}</div>
            <div class="probability-score">Caries Probability: {caries_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence > 0.8:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🦷 Excellent Oral Health</h4>
                <p><strong>Recommendation:</strong> Continue regular oral hygiene routine</p>
                <p><em>Maintain current dental care practices</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🔍 Good Oral Health</h4>
                <p><strong>Recommendation:</strong> Maintain oral health with regular checkups</p>
                <p><em>Continue preventive care routine</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Enhanced visualization with organized layout
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    
    if attention_map is not None:
        st.markdown("### 🔬 **Detailed AI Analysis**")
        
        # Organized visualization container
        st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
        
        # Create professional visualization with dark theme
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#374151')
        
        # Original image with dark theme styling
        ax1.imshow(image)
        ax1.set_title('Original X-ray Image', fontsize=16, fontweight='bold', pad=25, color='#e2e8f0')
        ax1.axis('off')
        ax1.set_facecolor('#374151')
        
        # Attention overlay with dark theme visualization
        ax2.imshow(image, alpha=0.85)
        im = ax2.imshow(attention_map, cmap='hot', alpha=0.65)
        ax2.set_title('AI Focus Areas (Heat Map)', fontsize=16, fontweight='bold', pad=25, color='#e2e8f0')
        ax2.axis('off')
        ax2.set_facecolor('#374151')
        
        # Enhanced colorbar with dark theme
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8, aspect=20)
        cbar.set_label('Attention Intensity', fontsize=12, color='#e2e8f0')
        cbar.ax.tick_params(labelsize=10, colors='#e2e8f0')
        cbar.ax.set_facecolor('#4b5563')
        
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Organized metrics display
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 **Analysis Metrics**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🎯 Prediction Confidence", 
                value=f"{confidence:.1%}",
                help="AI model's confidence in the prediction"
            )
        with col2:
            st.metric(
                label="🦷 Caries Probability", 
                value=f"{caries_prob:.1%}",
                help="Probability of caries presence"
            )
        with col3:
            attention_strength = np.mean(attention_map) if attention_map is not None else 0
            st.metric(
                label="🔬 Attention Strength", 
                value=f"{attention_strength:.3f}",
                help="Average attention map intensity"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.markdown("### 📷 **X-ray Analysis**")
        
        st.markdown('<div class="visualization-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Dental X-ray", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Basic metrics for cases without attention map
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        st.markdown("#### 📊 **Analysis Metrics**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="🎯 Prediction Confidence", 
                value=f"{confidence:.1%}",
                help="AI model's confidence in the prediction"
            )
        with col2:
            st.metric(
                label="🦷 Caries Probability", 
                value=f"{caries_prob:.1%}",
                help="Probability of caries presence"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def reset_app():
    """Reset the app state to allow new analysis"""
    st.session_state.show_results = False
    st.session_state.uploaded_file_key += 1
    st.rerun()

def main():
    # Professional Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🦷 AI Dental Caries Detection</div>
        <div class="hero-subtitle">Advanced Machine Learning for Dental Diagnostics</div>
        <div class="hero-description">
            Leverage cutting-edge artificial intelligence to detect dental caries with high precision. 
            Our advanced deep learning models provide instant analysis with confidence scoring and attention mapping.
        </div>
        <br>
        <div>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">Computer Vision</span>
            <span class="tech-badge">SegFormer</span>
            <span class="tech-badge">VGG16</span>
            <span class="tech-badge">Semantic Segmentation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 🧠 **Select AI Model**")
    model_type = st.selectbox(
        "Choose AI model for analysis",
        options=["segformer", "vgg16"],
        format_func=lambda x: "SegFormer (Semantic Segmentation)" if x == "segformer" else "VGG16 (Classification)",
        help="SegFormer provides pixel-level segmentation, VGG16 provides classification with attention"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load model with professional error handling
    try:
        with st.spinner(f"🤖 Loading {model_type.upper()} AI model..."):
            model, device = load_model(model_type)
        st.success(f"✅ {model_type.upper()} AI model loaded successfully")
    except Exception as e:
        st.error(f"❌ Could not load {model_type.upper()} model. Please check model file.")
        st.exception(e)
        return
    
    # Professional Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 **Upload Dental X-ray**")
    st.markdown("*Supported formats: PNG, JPG, JPEG, BMP, TIFF*")
    
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a clear dental X-ray image for AI analysis",
        key=f"file_uploader_{st.session_state.uploaded_file_key}"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file and not st.session_state.show_results:
        image = Image.open(uploaded_file)
        
        # Enhanced image info display
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.markdown(f"""
        <div class="image-info">
            <strong>📷 Image Information</strong><br>
            <strong>File:</strong> {uploaded_file.name}<br>
            <strong>Dimensions:</strong> {image.size[0]} × {image.size[1]} pixels<br>
            <strong>Size:</strong> {file_size_kb:.1f} KB<br>
            <strong>Format:</strong> {image.format}
        </div>
        """, unsafe_allow_html=True)
        
        # Automatic AI Analysis
        st.markdown("### 🔍 **AI Analysis**")
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Preprocessing image...")
            progress_bar.progress(25)
            
            status_text.text(f"🧠 Running {model_type.upper()} AI inference...")
            progress_bar.progress(50)
            
            predicted_class, confidence, caries_prob, attention_map = analyze_image(
                model, image, device, model_type=model_type
            )
            
            status_text.text("📊 Generating results...")
            progress_bar.progress(75)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show results and mark as shown
            show_results(np.array(image), predicted_class, confidence, caries_prob, attention_map)
            st.session_state.show_results = True
            
        except Exception as e:
            st.error("❌ Error during analysis")
            st.exception(e)
    
    # Enhanced reset section - only show when results are displayed
    if st.session_state.show_results:
        st.markdown('<div class="reset-section">', unsafe_allow_html=True)
        st.markdown("### 🔄 **New Analysis**")
        st.markdown("Ready to analyze another X-ray? Click below to start fresh.")
        
        if st.button("🆕 Analyze New Image", type="primary", use_container_width=True):
            reset_app()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("""
    <div class="footer-section">
        <h4>⚠️ Medical Disclaimer</h4>
        <p><strong>This AI tool is designed for educational and research purposes only.</strong></p>
        <p>Results should not replace professional dental examination or diagnosis. 
        Always consult with a qualified dentist for accurate diagnosis and treatment planning.</p>
        <br>
        <p style="font-size: 0.9rem; opacity: 0.7; color: #64748b;">
            Powered by PyTorch • SegFormer/VGG16 Architecture • Semantic Segmentation
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
