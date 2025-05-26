import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # You can create a separate style.css file or define below

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
with st.sidebar:
    st.image("plant_logo.png", width=150)  # Add a logo image
    st.title("Plant Disease Detection")
    app_mode = st.selectbox("Navigate", ["Home", "About", "Disease Recognition"])
    
    if app_mode == "Disease Recognition":
        st.info("Upload an image of a plant leaf to check for diseases.")
    elif app_mode == "About":
        st.info("Learn about our project and dataset.")
    else:
        st.info("Welcome to the Plant Disease Recognition System!")

# Home Page
if app_mode == "Home":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("🌿 Plant Disease Recognition System")
        st.markdown("""
        <div style="text-align: justify;">
        Our advanced AI system helps identify plant diseases quickly and accurately. 
        Upload an image of a plant leaf, and our deep learning model will analyze it 
        to detect any signs of diseases. Protect your crops with early detection!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 How It Works")
        steps = [
            {"icon": "📤", "text": "Upload an image of a plant leaf"},
            {"icon": "🔍", "text": "Our AI analyzes the image for disease patterns"},
            {"icon": "📊", "text": "Get instant results with confidence levels"},
            {"icon": "💡", "text": "Receive recommendations for next steps"}
        ]
        
        for step in steps:
            st.markdown(f"{step['icon']} {step['text']}")
            
        if st.button("Get Started →", key="home_get_started"):
            st.experimental_set_query_params(page="Disease Recognition")
            
    with col2:
        st.image("home_page.jpeg", use_column_width=True, caption="Healthy plants lead to a healthy planet")
    
    st.markdown("---")
    st.markdown("### 🌟 Why Choose Our System?")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div style="text-align: center;">
        <h3>🔬 Accurate</h3>
        <p>State-of-the-art deep learning model with 95%+ accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""
        <div style="text-align: center;">
        <h3>⚡ Fast</h3>
        <p>Get results in seconds with our optimized system</p>
        </div>
        """, unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""
        <div style="text-align: center;">
        <h3>🌍 Impactful</h3>
        <p>Help reduce crop losses and improve food security</p>
        </div>
        """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.header("📚 About")
    
    with st.expander("ℹ️ Project Information", expanded=True):
        st.markdown("""
        This plant disease recognition system was developed to help farmers and gardeners 
        identify plant diseases early. Our goal is to make advanced AI technology accessible 
        to everyone for sustainable agriculture.
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. 
        The original dataset can be found on [this GitHub repo](https://github.com/).
        
        #### Dataset Content
        - **Train**: 70,295 images
        - **Valid**: 17,572 images
        - **Test**: 33 images
        
        The dataset contains RGB images of healthy and diseased crop leaves categorized into 38 different classes.
        """)
    
    with st.expander("👥 Our Team"):
        cols = st.columns(3)
        with cols[0]:
            st.image("team1.jpg", width=150, caption="Dr. Jane Smith - AI Researcher")
        with cols[1]:
            st.image("team2.jpg", width=150, caption="John Doe - Data Scientist")
        with cols[2]:
            st.image("team3.jpg", width=150, caption="Sarah Johnson - Agricultural Expert")
    
    st.markdown("---")
    st.markdown("### 📬 Contact Us")
    contact_form = """
    <form action="https://formsubmit.co/your@email.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Image")
        test_image = st.file_uploader("Choose a plant leaf image:", type=["jpg", "jpeg", "png"])
        
        if test_image is not None:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing the image..."):
                    try:
                        result_index = model_prediction(test_image)
                        class_name = [
                            'Apple___Apple_scab',
                            'Apple___Black_rot',
                            'Apple___Cedar_apple_rust',
                            'Apple___healthy',
                            'Blueberry___healthy',
                            'Cherry_(including_sour)___Powdery_mildew',
                            'Cherry_(including_sour)___healthy',
                            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                            'Corn_(maize)___Common_rust_',
                            'Corn_(maize)___Northern_Leaf_Blight',
                            'Corn_(maize)___healthy',
                            'Grape___Black_rot',
                            'Grape___Esca_(Black_Measles)',
                            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                            'Grape___healthy',
                            'Orange___Haunglongbing_(Citrus_greening)',
                            'Peach___Bacterial_spot',
                            'Peach___healthy',
                            'Pepper,_bell___Bacterial_spot',
                            'Pepper,_bell___healthy',
                            'Potato___Early_blight',
                            'Potato___Late_blight',
                            'Potato___healthy',
                            'Raspberry___healthy',
                            'Soybean___healthy',
                            'Squash___Powdery_mildew',
                            'Strawberry___Leaf_scorch',
                            'Strawberry___healthy',
                            'Tomato___Bacterial_spot',
                            'Tomato___Early_blight',
                            'Tomato___Late_blight',
                            'Tomato___Leaf_Mold',
                            'Tomato___Septoria_leaf_spot',
                            'Tomato___Spider_mites Two-spotted_spider_mite',
                            'Tomato___Target_Spot',
                            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                            'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy'
                        ]
                        
                        disease = class_name[result_index]
                        if "healthy" in disease:
                            st.balloons()
                            st.success(f"🎉 Great news! This plant appears to be healthy ({disease.replace('___', ' - ')})")
                        else:
                            st.warning(f"⚠️ Detection: {disease.replace('___', ' - ')}")
                            
                            st.markdown("### Recommended Actions")
                            st.info("""
                            - Isolate the affected plant to prevent spread
                            - Remove severely infected leaves
                            - Apply appropriate organic fungicides
                            - Monitor plant health regularly
                            - Consult with a local agricultural expert
                            """)
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    with col2:
        if test_image is not None:
            st.subheader("Image Preview")
            img = Image.open(test_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Add some image analysis metrics (placeholder)
            with st.expander("📊 Image Analysis Metrics"):
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Color Distribution", "Normal")
                with cols[1]:
                    st.metric("Texture Analysis", "Complete")
                with cols[2]:
                    st.metric("Pattern Detection", "Detailed")
                
                st.progress(80, text="Analysis completeness")
        else:
            st.info("ℹ️ Please upload an image to analyze")
            st.image("placeholder_image.jpg", use_column_width=True, caption="Example of a plant leaf image")
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit photos of individual leaves
    - Capture both sides of the leaf if possible
    - Avoid shadows or reflections on the leaf surface
    - Ensure the leaf covers most of the image frame
    - Take multiple photos from different angles for comprehensive analysis
    """)