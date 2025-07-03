# 🖼 GPU Image Enhancer (OpenCL)

This project performs GPU-accelerated image enhancement using **OpenCL**. It applies a two-stage pipeline:
1. **Gaussian Blur** – to reduce noise and smooth the image.
2. **Logarithmic Tone Mapping** – to enhance contrast and dynamic range.

---

## 📂 Project Structure
---
gpu_image_enhancer/
│
├── images/
│ ├── input.png # Sample input image
│ └── output.png # Output after processing
│
├── kernels/
│ └── filters.cl # OpenCL C kernels (Gaussian blur + tone mapping)
│
├── src/
│ ├── main.py # CLI entry point to run the image processor
│ ├── image_processor.py # Core processing logic (GPU pipeline)
│ ├── opencl_setup.py # Platform, device, and program setup
│ └── utils.py # Image loading/saving helper functions
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Requirements

- Python 3.8+
- Conda (recommended) or pip
- OpenCL runtime (PoCL or vendor-specific)
- A device with OpenCL support (e.g., CPU with PoCL, or GPU)

Install dependencies:

```bash
# Create and activate conda environment
conda create -n openclenv python=3.10 -c conda-forge
conda activate openclenv

# Install dependencies
pip install -r requirements.txt
python src/main.py --input images/input.png --output images/output.png --luminance 5.0
