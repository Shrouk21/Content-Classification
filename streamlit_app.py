import pandas as pd
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from distilbert import DistilBertLoRATrainer
from augmentation import DataAugmentor
import io
import base64
import nltk
import streamlit as st

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.caption_model = None
    st.session_state.feature_extractor = None
    st.session_state.tokenizer = None
    st.session_state.trainer = None

@st.cache_resource
def load_models():
    """Load lightweight ViT-GPT2 model and DistilBERT trainer"""
    try:
        # Load lightweight ViT-GPT2 model for image captioning (~1GB)
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        
        caption_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load data and prepare DistilBERT model
        df = pd.read_csv("cellula toxic data.csv")
        augmentor = DataAugmentor()
        aug_df = augmentor.augment_df(df, method="synonym")
        
        trainer = DistilBertLoRATrainer(num_labels=aug_df['Toxic Category'].nunique())
        trainer.prepare_data(aug_df, col_a="image descriptions", col_b="query", label_col="Toxic Category")
        trainer.load_best_model(path='best_model.pt')
        
        return caption_model, feature_extractor, tokenizer, trainer, aug_df['Toxic Category'].unique()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def generate_image_caption(image, model, feature_extractor, tokenizer):
    """Generate caption for uploaded image using ViT-GPT2"""
    try:
        # Process image
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        
        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values, 
                max_length=50, 
                num_beams=1,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
            caption_model, feature_extractor, tokenizer, trainer, categories = load_models()
            
            if caption_model and feature_extractor and tokenizer and trainer:
                st.session_state.caption_model = caption_model
                st.session_state.feature_extractor = feature_extractor
                st.session_state.tokenizer = tokenizer
                st.session_state.trainer = trainer
                st.session_state.categories = categories
                st.session_state.model_loaded = True
                st.success("Models loaded successfully!")
            else:
                st.error("Failed to load models. Please check your model files.")
                return
    
    # Input Section
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
                        st.session_state.caption_model,
                        st.session_state.feature_extractor,
                        st.session_state.tokenizer
                    )
                
                if caption:
                    st.success("Caption generated successfully!")
                    st.info(f"**Generated Caption:** {caption}")
                    text_to_classify = caption
                else:
                    st.error("Failed to generate caption.")
            else:
                st.error(result)

    st.markdown("---")  # Separator between sections

    # Classification Results Section
    st.header("🔍 Classification Results")
    
    if text_to_classify:
        with st.spinner("Classifying content..."):
            try:
                # Get prediction with confidence scores
                prediction, confidence = st.session_state.trainer.predict(text_a=text_to_classify)
                
                # Display results with confidence
                st.subheader("Classification Result")
                
                # Create two columns for prediction and confidence
                pred_col, conf_col = st.columns([3, 1])
                
                with pred_col:
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
                
                with conf_col:
                    # Display confidence as percentage with progress bar
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.progress(float(confidence))

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
                        st.write("1. Image is processed using lightweight ViT-GPT2 model (~1GB)")
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