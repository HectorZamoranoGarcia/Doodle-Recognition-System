import gradio as gr
import numpy as np
from PIL import ImageOps
import os
import sys


# Configuration and Backend Import

try:
    import backend_math as model
except ImportError:
    try:
        import LogisticRegression as model
    except ImportError:
        print("Error: Could not find 'backend_math.py' or 'LogisticRegression.py'.")
        sys.exit(1)

# Map numeric labels to readable English names
# This is vital for the demo to be user-friendly.
DOODLE_MAP = {
    0: "🙂 Smiley Face",
    1: "❤️ Heart",
    2: "⭐ Star",
    3: "⚡ Lightning",
    4: "🌙 Moon",
    5: "☁️ Cloud",
    6: "🌲 Tree",
    7: "🏠 House",
    8: "☹️ Sad Face",
    9: "☀️ Sun"
}


# Image Preprocessing (16x16 Pipeline)
def preprocess_image(image):
    """
    Converts the user's drawing into a normalized feature vector
    of 16x16 blocks (256 dimensions), matching the training data.
    """
    # 1. Convert to grayscale
    image = image.convert("L")

    # 2. Invert colors (Gradio provides white background, model expects black)
    inverted_image = ImageOps.invert(image)

    # 3. Crop to content (Bounding Box)
    bbox = inverted_image.getbbox()
    if bbox is None:
        return None
    cropped = inverted_image.crop(bbox)

    # 4. Resize to standard 32x32
    resized = cropped.resize((32, 32))
    x = np.array(resized, dtype=np.uint8)

    # 5. Binarize (Pure Black/White)
    x = np.where(x > 128, 1, 0)

    # 6. Downsampling to 16x16 (Feature Engineering)
    xb = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            xb[i, j] = np.sum(x[i*2:(i+1)*2, j*2:(j+1)*2])

    # 7. Flatten and normalize (divide by 4.0 because it's the max value of a 2x2 block)
    return xb.reshape(1, -1) / 4.0


# Prediction Logic
def classify(canvas, W, classifier_func):
    # Handle Gradio's new dictionary format (if using new Paint component)
    if isinstance(canvas, dict):
        image = canvas.get("composite")
    else:
        image = canvas

    if image is None:
        return "Empty canvas"

    # Preprocess
    x = preprocess_image(image)
    if x is None:
        return "Draw something first"

    # Predict
    prediction_index = classifier_func(x, W)[0] # Returns array, take 1st element

    # Translate index to name
    prediction_name = DOODLE_MAP.get(int(prediction_index), "Unknown")

    return f"It looks like a {prediction_name}!"


# Graphic Interface (Gradio)
def create_interface(W, classifier_func):

    # Wrapper to pass fixed weights to the classify function
    def classify_wrapper(image):
        return classify(image, W, classifier_func)

    with gr.Blocks(title="AI Doodle Recognizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI Doodle Recognizer")
        gr.Markdown("""
        This neural network has been trained **from scratch** (no Deep Learning libraries)
        to recognize your hand-drawn doodles.

        Try drawing: **Sun, Moon, Tree, Heart, Star, etc.**
        """)

        with gr.Row():
            with gr.Column():
                canvas = gr.Paint(
                    label="Draw here",
                    brush=gr.Brush(default_color="black", default_size=8),
                    type="pil",
                    height=300
                )
                classify_btn = gr.Button("🔮 Guess Doodle", variant="primary")

            with gr.Column():
                output = gr.Label(label="AI Prediction")

        # Events
        classify_btn.click(fn=classify_wrapper, inputs=canvas, outputs=output)

    return demo


# Main Entry Point
if __name__ == "__main__":
    # Expected path for your trained weights
    WEIGHTS_FILE = "models/weights_v1.npy"

    # 1. Load Weights
    if os.path.exists(WEIGHTS_FILE):
        print(f"Loading model from: {WEIGHTS_FILE}")
        W = np.load(WEIGHTS_FILE)
    elif os.path.exists("MyDigitsWeights.npy"):
        print("Warning: Using legacy weights found in root (MyDigitsWeights.npy)")
        W = np.load("MyDigitsWeights.npy")
    else:
        print(f"CRITICAL ERROR: Weights file not found at '{WEIGHTS_FILE}'.")
        print("   Please run the training script first (notebooks/03_model_training.ipynb)")
        print("   or ensure you saved the weights in the 'models/' directory.")
        sys.exit(1)

    # 2. Select backend classifier
    # Using Logistic Regression (Softmax)
    classifier = model.LogisticRegressionClassification

    # 3. Launch App
    demo = create_interface(W, classifier)
    demo.launch()
