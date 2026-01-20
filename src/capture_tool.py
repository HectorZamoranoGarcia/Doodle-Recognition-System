import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import os

class DigitCaptureApp:
    def __init__(self):
        self.demo = self.create_interface()
        self.X = np.array([])
        self.y = np.array([], dtype=np.int64)

        # Create data folder if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")


    # Utils
    def status(self, msg):
        counts = np.unique(self.y, return_counts=True)
        counts = ', '.join([f"'{c}': {v}" for c,v in zip(counts[0],counts[1])])
        return f'{msg}\n\nDataset Stats: ({counts})'

    def load_X_y(self, image_fn, label_fn, mode='loading'):
        if not image_fn or not label_fn:
            raise RuntimeError(f"Please provide filenames.")

        X = np.load(image_fn)
        # VALIDACIÓN 16x16 (256 píxeles)
        if X.dtype != np.float64 or len(X.shape) != 2 or X.shape[1] != 256 or X.shape[0] <= 0:
            raise RuntimeError(f'{image_fn} is not valid. Expected 256 features (16x16).')

        y = np.load(label_fn)
        if y.shape[0] != X.shape[0]:
            raise RuntimeError(f'Mismatch between labels and images.')
        return X, y


    # Preprocessing (16x16 Pipeline)
    def preprocess_image(self, image):
        image = image.convert("L")
        inverted_image = ImageOps.invert(image)
        bbox = inverted_image.getbbox()
        if bbox is None: return None

        cropped = inverted_image.crop(bbox)
        resized = cropped.resize((32, 32))
        x = np.array(resized, dtype=np.uint8)
        x = np.where(x > 128, 1, 0)


        # Downsampling a 16x16
        xb = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                xb[i, j] = np.sum(x[i*2:(i+1)*2, j*2:(j+1)*2])

        return xb.reshape(1, -1) / 4.0


    # Actions
    def save_image(self, canvas, label):
        if isinstance(canvas, dict): image = canvas.get("composite")
        else: image = canvas

        if image is None: return self.status("Empty canvas.")
        if not label or not label.strip().isdigit(): return self.status("Invalid label (must be 0-9).")

        label = int(label)
        x = self.preprocess_image(image)

        if self.X.size == 0: self.X = x
        else: self.X = np.vstack([self.X, x])
        self.y = np.append(self.y, label)

        return self.status(f"Saved class {label}")

    def save_dataset(self, image_fn, label_fn):
        np.save(image_fn, self.X)
        np.save(label_fn, self.y)
        return self.status(f"Saved to {image_fn} & {label_fn}")

    def load_dataset(self, image_fn, label_fn):
        try:
            self.X, self.y = self.load_X_y(image_fn, label_fn)
            return self.status(f"Loaded from {image_fn}")
        except Exception as e: return self.status(f"Error: {e}")

    # UI
    def create_interface(self):
        with gr.Blocks(title="Doodle Dataset Creator") as demo:
            gr.Markdown("## Doodle Dataset Creator")
            gr.Markdown("Draw shapes to build your custom dataset (16x16 resolution).")

            with gr.Row():
                with gr.Column():
                    canvas = gr.Paint(label="Draw Here", brush=gr.Brush(default_color="black", default_size=8), type="pil", height=250)
                    label_input = gr.Textbox(label="Class Label (0-9)", placeholder="e.g. 0 for Smiley, 1 for Heart...")
                    save_btn = gr.Button("Save Sample", variant="primary")

                with gr.Column():
                    output = gr.Textbox(label="Session Status", lines=6)

            save_btn.click(fn=self.save_image, inputs=[canvas, label_input], outputs=output)

            gr.Markdown("### Dataset Management")
            with gr.Row():
            # DEFAULT PATHS TO THE DATA FOLDER
                image_fn = gr.Textbox(label="Images Path", value="data/doodles_X.npy")
                label_fn = gr.Textbox(label="Labels Path", value="data/doodles_y.npy")

            with gr.Row():
                save_ds_btn = gr.Button("Save to Disk")
                load_ds_btn = gr.Button("Load from Disk")

            save_ds_btn.click(fn=self.save_dataset, inputs=[image_fn, label_fn], outputs=output)
            load_ds_btn.click(fn=self.load_dataset, inputs=[image_fn, label_fn], outputs=output)

        return demo

if __name__ == "__main__":
    app = DigitCaptureApp()
    app.demo.launch()
