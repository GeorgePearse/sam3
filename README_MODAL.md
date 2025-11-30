# Running SAM3 on Modal

This repository includes a script to run SAM3 inference using [Modal](https://modal.com/).

## Prerequisites

1.  **Modal Account**: Sign up at [modal.com](https://modal.com/).
2.  **Modal CLI**: Install and authenticate:
    ```bash
    pip install modal
    modal setup
    ```
3.  **Hugging Face Token**: SAM3 requires access to checkpoints on Hugging Face.
    *   Request access at [facebook/sam3](https://huggingface.co/facebook/sam3).
    *   Create a Modal secret named `huggingface-secret` with your token:
        ```bash
        modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
        ```

## Usage

Run the inference script:

```bash
modal run modal_app.py
```

You can customize the image and prompt:

```bash
modal run modal_app.py --image-path "assets/images/truck.jpg" --prompt "the truck"
```

## Details

The `modal_app.py` script:
*   Builds a Debian-based image with Python 3.12 and CUDA 12.6 support.
*   Installs PyTorch 2.7.0 and SAM3 dependencies.
*   Mounts the local `sam3` package and `assets` directory.
*   Runs the inference on an H100 GPU (configurable).
