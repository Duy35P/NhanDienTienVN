import os
import gradio as gr
import sys
from PIL import Image
import numpy as np
import cv2

# Reuse pipeline from Flask app which already loads both models and utilities
try:
    from app import process_image, EFF_CLASSES
except Exception as e:
    process_image = None
    EFF_CLASSES = None
    print("Warning: couldn't import process_image from app.py:", e)

# Nếu model trên HF Hub:
# model_path = hf_hub_download(repo_id="USERNAME/MODEL_REPO", filename="best.pt", repo_type="model", use_auth_token=os.getenv("HF_TOKEN"))

def run_pipeline_on_pil(pil_img):
    """Run the full pipeline from app.py on a PIL.Image and return (PIL result image, meta dict)."""
    if process_image is None:
        raise RuntimeError("Pipeline not available — ensure app.py can be imported and models load correctly.")

    # PIL -> RGB numpy -> BGR for OpenCV pipeline
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    result_bgr, detections, total_value = process_image(bgr)

    # Convert result back to PIL RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    meta = {
        "total_value": int(total_value),
        "total_bills": len(detections),
        "detections": detections,
    }
    return result_pil, meta


def predict_wrapper(img: Image.Image, mode: str):
    """Gradio wrapper: mode can be 'pipeline' or 'yolo-only'/'eff-only' (pipeline recommended)."""
    # For now we only implement pipeline which uses both YOLO+EfficientNet from app.py
    if mode != "pipeline":
        return None, {"error": "Only 'pipeline' mode implemented in this demo."}
    return run_pipeline_on_pil(img)


demo = gr.Interface(
    fn=predict_wrapper,
    inputs=[gr.Image(type="pil"), gr.Radio(choices=["pipeline"], value="pipeline", label="Mode")],
    outputs=[gr.Image(type="pil", label="Result Image"), gr.JSON(label="Metadata")],
    title="Currency detection + classification"
)

if __name__ == "__main__":
    # If an argument is provided, treat it as an image path and run prediction in CLI mode
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            sys.exit(2)
        pil = Image.open(img_path).convert("RGB")
        try:
            result_img, meta = run_pipeline_on_pil(pil)
        except Exception as exc:
            print("Error running pipeline:", exc)
            sys.exit(3)

        out_path = "result_" + os.path.basename(img_path)
        result_img.save(out_path)
        print("Saved result to", out_path)
        print("Meta:")
        print(meta)
    else:
        demo.launch(server_name="0.0.0.0", server_port=8080)