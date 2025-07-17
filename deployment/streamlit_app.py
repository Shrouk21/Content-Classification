import nltk
import streamlit as st
import pandas as pd
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from distilbert import DistilBertLoRATrainer
from augmentation import DataAugmentor
import io
import base64

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.blip_model = None
    st.session_state.blip_processor = None
    st.session_state.trainer = None

@st.cache_resource
def load_models():
    """Load BLIP model and DistilBERT trainer"""
    try:
        # Load BLIP model for image captioning
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Load data and prepare DistilBERT model
        df = pd.read_csv("cellula toxic data.csv")
        augmentor = DataAugmentor()
        aug_df = augmentor.augment_df(df, method="synonym")
        
        trainer = DistilBertLoRATrainer(num_labels=aug_df['Toxic Category'].nunique())
        trainer.prepare_data(aug_df, col_a="image descriptions", col_b="query", label_col="Toxic Category")
        trainer.load_best_model(path='best_model.pt')
        
        return blip_processor, blip_model, trainer, aug_df['Toxic Category'].unique()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def generate_image_caption(image, processor, model):
    """Generate caption for uploaded image using BLIP"""
    try:
        # Process image
        inputs = processor(image, return_tensors="pt")
        
        # Generate caption
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None

def validate_image(uploaded_file):
    """Validate uploaded image file"""
    try:
        # Check file size (limit to 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File size too large (max 10MB)"
        
        # Check file type
        if uploaded_file.type not in ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']:
            return False, "Unsupported file type. Please upload JPEG, PNG, or GIF."
        
        # Try to open image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return True, image
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def main():
    st.set_page_config(
        page_title="Toxic Content Moderation System",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Toxic Content Moderation System")
    st.markdown("---")
    
    # Load models
    if not st.session_state.model_loaded:
        with st.spinner("Loading models... This may take a moment."):
            processor, blip_model, trainer, categories = load_models()
            
            if processor and blip_model and trainer:
                st.session_state.blip_processor = processor
                st.session_state.blip_model = blip_model
                st.session_state.trainer = trainer
                st.session_state.categories = categories
                st.session_state.model_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please check your model files.")
                return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Options")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "Image Upload"],
            horizontal=True
        )
        
        text_to_classify = ""
        
        if input_method == "Text Input":
            st.subheader("Enter Text")
            text_to_classify = st.text_area(
                "Enter your text for classification:",
                height=150,
                placeholder="Type or paste your text here..."
            )
            
        elif input_method == "Image Upload":
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif'],
                help="Supported formats: PNG, JPG, JPEG, GIF (max 10MB)"
            )
            
            if uploaded_file is not None:
                # Validate image
                is_valid, result = validate_image(uploaded_file)
                
                if is_valid:
                    image = result
                    
                    # Display uploaded image
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Generate caption
                    with st.spinner("Generating image caption..."):
                        caption = generate_image_caption(
                            image, 
                            st.session_state.blip_processor, 
                            st.session_state.blip_model
                        )
                    
                    if caption:
                        st.success("Caption generated successfully!")
                        st.info(f"**Generated Caption:** {caption}")
                        text_to_classify = caption
                    else:
                        st.error("Failed to generate caption.")
                else:
                    st.error(result)
    
    with col2:
        st.header("🔍 Classification Results")
        
        if text_to_classify:
            with st.spinner("Classifying content..."):
                try:
                    # Get prediction
                    prediction = st.session_state.trainer.predict(text_a=text_to_classify)
                    
                    # Display results
                    st.subheader("Classification Result")
                    
                    # Create colored alert based on toxicity category
                    if prediction.lower() == 'safe':
                        st.success(f"✅ **Content is Safe**: {prediction}")
                    elif prediction.lower() in ['child sexual exploitation', 'sex-related crimes']:
                        st.error(f"🚨 **CRITICAL THREAT - {prediction}**: Content flagged for immediate review")
                    elif prediction.lower() in ['violent crimes', 'suicide & self-harm']:
                        st.error(f"🚨 **HIGH RISK - {prediction}**: Dangerous content detected")
                    elif prediction.lower() in ['unsafe', 'non-violent crimes']:
                        st.warning(f"⚠️ **MODERATE RISK - {prediction}**: Content requires attention")
                    elif prediction.lower() == 'elections':
                        st.info(f"📊 **POLITICAL CONTENT - {prediction}**: Election-related content detected")
                    elif prediction.lower() == 'unknown s-type':
                        st.warning(f"❓ **UNKNOWN TYPE - {prediction}**: Content classification uncertain")
                    else:
                        st.info(f"ℹ️ **Classification**: {prediction}")
                    
                    # Show input text for reference
                    st.subheader("Analyzed Text")
                    st.text_area("", value=text_to_classify, height=100, disabled=True)
                    
                    # Additional info
                    with st.expander("ℹ️ More Information"):
                        st.write("**Available Categories:**")
                        for category in st.session_state.categories:
                            st.write(f"• {category}")
                        
                        st.write("**How it works:**")
                        if input_method == "Image Upload":
                            st.write("1. Image is processed using BLIP model to generate a text caption")
                            st.write("2. Caption is analyzed by DistilBERT model for toxicity classification")
                        else:
                            st.write("1. Text is directly analyzed by DistilBERT model for toxicity classification")
                        
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        else:
            st.info("👆 Please enter text or upload an image to classify.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This system uses AI models for content moderation. "
        "Results should be reviewed by human moderators for final decisions."
    )

if __name__ == "__main__":
    main()