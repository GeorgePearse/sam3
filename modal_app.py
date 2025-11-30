import os
import sys
import modal

# Define the image with necessary dependencies
# We use a base image with Python 3.12 and install system dependencies for OpenCV
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .run_commands(
        # Install PyTorch 2.7.0 and related libraries from the specified index
        "uv pip install --system torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
        # Install other dependencies listed in pyproject.toml
        "uv pip install --system timm>=1.0.17 numpy==1.26 tqdm ftfy==6.1.1 regex iopath>=0.1.10 typing_extensions huggingface_hub opencv-python pillow matplotlib decord scikit-image scikit-learn",
    )
)

app = modal.App("sam3-inference", image=image)

# Mount the sam3 package and assets to preserve the directory structure
# /root/sam3 and /root/assets
sam3_mount = modal.Mount.from_local_dir(
    "sam3",
    remote_path="/root/sam3",
    condition=lambda path: not any(
        p in path for p in ["__pycache__", ".git", ".venv", "egg-info"]
    ),
)

assets_mount = modal.Mount.from_local_dir(
    "assets",
    remote_path="/root/assets",
    condition=lambda path: not any(p in path for p in ["__pycache__"]),
)


@app.function(
    gpu="H100",  # SAM3 is a large model, suggest using a capable GPU
    mounts=[sam3_mount, assets_mount],
    secrets=[
        modal.Secret.from_name("huggingface-secret")
    ],  # Requires a secret named 'huggingface-secret' with HF_TOKEN
    timeout=600,
)
def inference(image_path: str, prompt: str):
    """
    Runs SAM3 inference on the provided image and prompt.
    """
    import sys

    # Add /root to python path so 'sam3' package can be imported
    if "/root" not in sys.path:
        sys.path.append("/root")

    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Ensure HF_TOKEN is available
    if "HF_TOKEN" not in os.environ:
        print(
            "Warning: HF_TOKEN not found in environment. Model download might fail if you don't have access."
        )

    print(f"Building SAM3 Image Model...")
    # This will download the checkpoint from HF if not cached
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # Resolve image path
    full_path = os.path.join("/root", image_path)
    if not os.path.exists(full_path):
        # Fallback to absolute path or relative to current if not found in /root
        if os.path.exists(image_path):
            full_path = image_path
        else:
            raise FileNotFoundError(f"Image not found at {full_path} or {image_path}")

    print(f"Loading image from {full_path}...")
    image_pil = Image.open(full_path).convert("RGB")

    print(f"Running inference with prompt: '{prompt}'...")
    inference_state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]
    boxes = output["boxes"]

    print(f"Inference complete. Found {len(masks)} masks.")

    # Return serializable results
    return {
        "count": len(masks),
        "scores": scores.tolist(),
        "boxes": boxes.tolist(),
        # We generally avoid returning heavy mask arrays directly in simple examples,
        # but could return shapes or encoded masks if needed.
    }


@app.local_entrypoint()
def main(
    image_path: str = "assets/images/test_image.jpg",
    prompt: str = "the leftmost child wearing blue vest",
):
    print(f"Starting inference on Modal...")
    try:
        result = inference.remote(image_path, prompt)
        print("Success!")
        print(f"Masks found: {result['count']}")
        print(f"Scores: {result['scores']}")
        print(f"Boxes: {result['boxes']}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "\nNote: Ensure you have set the 'huggingface-secret' in Modal with your HF_TOKEN."
        )
        print("Run: modal secret create huggingface-secret HF_TOKEN=your_token_here")
