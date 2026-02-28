import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("Best_Model_On_Partial.keras")

# Class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return f"🧠 Prediction: {class_labels[predicted_class]} (Confidence: {confidence:.2f}%)"

# Customizing Gradio UI
custom_css = """
    body {background-color: #1A1F3B; color: #E0E0E0; font-family: Arial, sans-serif;}
    .gradio-container {max-width: 800px; margin: auto; text-align: center;}
    .gr-button {background-color: #007BFF !important; color: white !important; border-radius: 8px;}
    .gr-box {background-color: #2C3E50; padding: 10px; border-radius: 10px;}
"""

description = """
🧠 **Brain Tumor Detector**  
Upload an MRI scan to classify brain tumors using deep learning.  
💡 **Supports:** Glioma | Meningioma | No Tumor | Pituitary  
🚀 **Fast & Accurate AI Model**
"""

with gr.Blocks() as interface:
    gr.HTML("<img src='Figure 2025-01-07 031757 (8).png' style='width:100px; position:absolute; top:10px; left:10px;'>")
    gr.Markdown("## Brain Tumor Detection 🧠")
    image = gr.Image(type="pil", label="Upload Brain MRI")
    output = gr.Textbox(label="Prediction")
    btn = gr.Button("Predict")
    
    btn.click(predict, inputs=image, outputs=output)
    
# UI Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Brain MRI"),
    outputs=gr.Textbox(label="Prediction"),
    title="Brain Tumor Detection 🧠",
    description="Upload an MRI scan to classify brain tumors using deep learning.",
    theme="default",
    css=custom_css,
    examples=["/mnt/data/2.webp"]  # Use the uploaded image as an example
)

# Launch the app
interface.launch()

