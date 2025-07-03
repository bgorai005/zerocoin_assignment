#  GPU Image Enhancement using OpenCL

---
##  Overview

This project demonstrates GPU-accelerated image enhancement using **OpenCL** with **PyOpenCL** in Python. The image is processed through **two OpenCL kernels**:

1. **3x3 Gaussian Blur:** Smooths the image and reduces noise per RGB channel.
2. **Logarithmic Tone Mapping:** Enhances contrast by applying a perceptually motivated tone-mapping function on luminance.

The final image is saved after both stages, showcasing the power of parallel GPU execution for image processing.

---

##  Repository  Structure
```
gpu_image_enhancer/
├── images/
│ ├── input.png # Sample input image
│ └── output.png # Output after processing
│
├── kernels/
│ └── filters.cl # OpenCL C kernels (Gaussian blur + tone mapping)
├── src/
│ ├── image_processor.py # Core processing logic (GPU pipeline)
│ ├── opencl_setup.py # Platform, device, and program setup
│ └── utils.py # Image loading/saving helper functions
├── main.py # CLI entry point to run the image processor
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```

---

## 💻 Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- OpenCL-capable system (CPU or GPU with drivers installed)
- Conda or pip
- [PoCL (Portable OpenCL)](http://portablecl.org/) if no vendor OpenCL driver is present

### 📦 Setup (via Conda – Recommended)

```bash
# Step 1: Create virtual environment
conda create -n openclenv python=3.10 -c conda-forge
conda activate openclenv

# Step 2: Install dependencies
pip install -r requirements.txt
```
# For Run
```bash
python src/main.py --input images/input.png --output images/output.png --luminance 5.0
```





