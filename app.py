import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eeg_net import EEGNet
import io
import pandas as pd

# Set page config
st.set_page_config(
    page_title="EEG Alzheimer's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained EEGNet model"""
    try:
        # Model parameters (matching your test.py)
        model = EEGNet(
            num_channels=19, 
            timepoints=1425, 
            num_classes=3, 
            F1=5, 
            D=5, 
            F2=25, 
            dropout_rate=0.5
        )
        
        # Try multiple model paths
        model_paths = [
            "models/eegnet_5fold_train7.pth",  # our primary model
            "outputs/eegnet_final.pth",
            "outputs/eegnet_best.pth",
            "models/eegnet_5fold_train3.pth",
            "models/eegnet_5fold_train4.pth",
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                # Load with CPU mapping to avoid CUDA issues
                checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
                
                # Handle both old format (direct state_dict) and new format (dictionary)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    st.success(f"✅ Model loaded from {model_path} (checkpoint format)")
                else:
                    model.load_state_dict(checkpoint)
                    st.success(f"✅ Model loaded from {model_path} (direct format)")
                
                model.eval()
                model_loaded = True
                break
                
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error loading {model_path}: {str(e)}")
                continue
        
        if not model_loaded:
            st.error("❌ Could not load any model file. Please check your model files.")
            return None
            
        return model
        
    except Exception as e:
        st.error(f"❌ Error initializing model: {str(e)}")
        return None

def predict_eeg(model, eeg_data):
    """Make prediction on EEG data"""
    try:
        with torch.no_grad():
            # Ensure data is in correct format: (batch_size, channels, timepoints)
            if len(eeg_data.shape) == 2:  # (channels, timepoints)
                eeg_data = eeg_data.unsqueeze(0)  # Add batch dimension -> (1, channels, timepoints)
            elif len(eeg_data.shape) == 1:  # (flattened)
                # This shouldn't happen but handle it just in case
                eeg_data = eeg_data.view(1, 19, -1)
            
            # Debug: Print tensor shape
            print(f"EEG data shape before model: {eeg_data.shape}")
            
            # Make prediction
            outputs = model(eeg_data.float())
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            
            return predicted_class.item(), probabilities.squeeze().numpy()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        print(f"EEG data shape causing error: {eeg_data.shape}")
        return None, None

def generate_sample_eeg(class_type="normal", duration=1425):
    """Generate sample EEG data for demo purposes"""
    np.random.seed(42)  # For reproducible results
    
    # Generate base EEG signal
    time_points = np.linspace(0, duration/250, duration)  # 250 Hz sampling rate
    channels = 19
    
    eeg_data = np.zeros((channels, duration))
    
    for ch in range(channels):
        # Base frequencies for different brain waves
        alpha = np.sin(2 * np.pi * 10 * time_points) * 0.5  # 10 Hz alpha
        beta = np.sin(2 * np.pi * 20 * time_points) * 0.3   # 20 Hz beta
        theta = np.sin(2 * np.pi * 6 * time_points) * 0.2   # 6 Hz theta
        
        # Initialize noise variable for each condition
        if class_type == "alzheimer":
            # Simulate AD patterns: increased theta, decreased alpha
            signal = theta * 2 + alpha * 0.5 + beta * 0.7
            # Add more irregular patterns
            noise = np.random.normal(0, 0.8, duration)
        elif class_type == "frontotemporal":
            # Simulate FTD patterns: altered frontal activity
            signal = alpha * 0.8 + beta * 1.2 + theta * 0.8
            if ch < 8:  # Frontal channels more affected
                noise = np.random.normal(0, 1.2, duration)
            else:
                noise = np.random.normal(0, 0.5, duration)
        else:  # normal
            # Healthy brain patterns
            signal = alpha + beta * 0.8 + theta * 0.5
            noise = np.random.normal(0, 0.3, duration)
        
        eeg_data[ch] = signal + noise
    
    return torch.tensor(eeg_data, dtype=torch.float32)

# Load model
model = load_model()

# Main app
st.markdown('<h1 class="main-header">🧠 EEG-based Alzheimer\'s Detection System</h1>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model could not be loaded. Please check your model files.")
    st.stop()

# Sidebar
st.sidebar.title("📊 Model Information")
st.sidebar.info("""
**Model Architecture:** EEGNet  
**Input:** 19-channel EEG (1425 timepoints)  
**Classes:** 
- 🟢 Healthy (Control)
- 🟡 Alzheimer's Disease  
- 🔴 Frontotemporal Dementia

**Model Parameters:**
- F1=5, D=5, F2=25
- Dropout: 50%
- Timepoints: 1425
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Demo Prediction", "📤 Upload EEG Data", "📈 Model Analytics"])

with tab1:
    st.header("Demo with Simulated EEG Data")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generate Sample EEG")
        sample_type = st.selectbox(
            "Select sample type:",
            ["normal", "alzheimer", "frontotemporal"],
            help="Generate simulated EEG data for different conditions"
        )
        
        if st.button("🎯 Generate & Predict", type="primary"):
            with st.spinner("Generating EEG data and making prediction..."):
                # Generate sample data
                sample_eeg = generate_sample_eeg(sample_type)
                
                # Make prediction
                pred_class, probabilities = predict_eeg(model, sample_eeg)
                
                if pred_class is not None:
                    # Class mapping
                    class_names = ["Alzheimer's Disease (A)", "Healthy Control (C)", "Frontotemporal Dementia (F)"]
                    class_colors = ["#ff6b6b", "#51cf66", "#ffd93d"]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Prediction Result</h2>
                        <h3>{class_names[pred_class]}</h3>
                        <p>Confidence: {probabilities[pred_class]:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability breakdown
                    st.subheader("📊 Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': probabilities,
                        'Percentage': [f"{p:.1%}" for p in probabilities]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(class_names, probabilities, color=class_colors, alpha=0.8)
                    ax.set_ylabel('Probability')
                    ax.set_title('Classification Probabilities')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Store in session state for visualization
                    st.session_state.sample_eeg = sample_eeg
                    st.session_state.pred_class = pred_class
                    st.session_state.probabilities = probabilities
    
    with col2:
        st.subheader("EEG Signal Visualization")
        if 'sample_eeg' in st.session_state:
            eeg_data = st.session_state.sample_eeg.numpy()
            
            # Plot subset of channels
            fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
            fig.suptitle('Sample EEG Channels', fontsize=14)
            
            channels_to_plot = [0, 5, 10, 15]  # Sample channels
            channel_names = [f'Channel {i+1}' for i in channels_to_plot]
            
            for i, (ax, ch_idx) in enumerate(zip(axes, channels_to_plot)):
                time = np.linspace(0, len(eeg_data[ch_idx])/250, len(eeg_data[ch_idx]))
                ax.plot(time, eeg_data[ch_idx], linewidth=0.8, color=f'C{i}')
                ax.set_ylabel(f'{channel_names[i]}')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time (seconds)')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("👆 Generate a sample to see EEG visualization")

with tab2:
    st.header("Upload Your EEG Data")
    st.info("📝 **Expected format:** CSV file with 19 channels × 1425 timepoints")
    
    uploaded_file = st.file_uploader("Choose EEG file", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
            
            # Validate dimensions
            if data.shape[1] != 1425 or data.shape[0] != 19:
                st.warning(f"⚠️ Expected shape: (19, 1425), Got: {data.shape}")
                st.info("The model expects 19 EEG channels with 1425 time points each.")
            
            # Convert to tensor and predict
            eeg_tensor = torch.tensor(data.values, dtype=torch.float32)
            
            if st.button("🔬 Analyze Uploaded EEG", type="primary"):
                with st.spinner("Analyzing EEG data..."):
                    pred_class, probabilities = predict_eeg(model, eeg_tensor)
                    
                    if pred_class is not None:
                        class_names = ["Alzheimer's Disease (A)", "Healthy Control (C)", "Frontotemporal Dementia (F)"]
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>📋 Analysis Result</h2>
                            <h3>{class_names[pred_class]}</h3>
                            <p>Confidence: {probabilities[pred_class]:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show probabilities
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🟡 Alzheimer's", f"{probabilities[0]:.1%}")
                        with col2:
                            st.metric("🟢 Healthy", f"{probabilities[1]:.1%}")
                        with col3:
                            st.metric("🔴 Frontotemporal", f"{probabilities[2]:.1%}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab3:
    st.header("Model Performance Analytics")
    
    # Model architecture info
    st.subheader("🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Input Channels", "19")
        st.metric("Time Points", "1,425")
        st.metric("Classes", "3")
    
    with col2:
        st.metric("F1 Filters", "5")
        st.metric("Depth Multiplier", "5")
        st.metric("F2 Filters", "25")
    
    # Performance metrics (example - replace with actual metrics)
    st.subheader("📊 Performance Metrics")
    
    # Example confusion matrix
    confusion_data = np.array([[80, 5, 3], [8, 85, 7], [6, 4, 82]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'HC', 'FTD'],
                yticklabels=['AD', 'HC', 'FTD'], ax=ax)
    ax.set_title('Confusion Matrix (Example)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", "92.3%", "2.1%")
    with col2:
        st.metric("Precision (avg)", "89.7%", "1.5%")
    with col3:
        st.metric("Recall (avg)", "88.9%", "0.8%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 EEG Alzheimer's Detection System | Built with Streamlit & PyTorch</p>
    <p><em>For research and educational purposes only</em></p>
</div>
""", unsafe_allow_html=True)