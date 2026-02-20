# 🧠 Getting Started: Implementation Guide

## 🛠️ Step-by-Step Deployment

<div align="center">

![Implementation](https://img.shields.io/badge/Implementation-Step--by--Step-blue?style=for-the-badge)
![Timeline](https://img.shields.io/badge/Timeline-4%20Weeks-yellowgreen?style=for-the-badge)
![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange?style=for-the-badge)

**[📋 Setup](01_COMPLETE_SETUP_GUIDE.md) • [🏗️ Architecture](02_PROJECT_ARCHITECTURE.md) • [🧪 Results](03_RESULTS_TESTBENCH.md)**

</div>

---

## 📋 Overview

New to this project? Start here. We'll walk you through everything in the right order.

## The Big Picture

You have an Alzheimer's classification model and want to run it on a PYNQ-ZU FPGA board. This guide tells you exactly what to do, in order.

```
Week 1: Setup          → Get hardware working
Week 2: Prepare Model  → Quantize and compile
Week 3: Deploy         → Move to PYNQ and test
Week 4: Run Live       → Webcam inference
```

## Week 1: Setup & Hardware

### Day 1-2: Get Your Board Ready

**You need:**
- PYNQ-ZU board
- 16GB microSD card
- 12V power supply
- Ethernet cable
- USB cable (for console)

**What to do:**
1. Go to [pynq.io](https://pynq.io) and download PYNQ 3.0 for ZU
2. Use [Balena Etcher](https://www.balena.io/etcher/) to flash the microSD card
3. Insert SD card into board
4. Connect power (wait for LEDs to blink)
5. Connect ethernet to your router

**Check it worked:**
```bash
# On your laptop/PC
ping pynq

# Should respond with something like:
# Reply from 192.168.1.100: bytes=32 time=5ms TTL=64
```

### Day 3: Access Jupyter Notebook

```
Open browser: http://pynq:9090
Username: xilinx
Password: xilinx
```

You should see the Jupyter interface. Congrats! Your board is alive.

**Pro tip**: Bookmark this page. You'll use it constantly.

### Day 4-5: Install Dependencies

Open a terminal in Jupyter (click "New" → "Terminal"):

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Vitis AI
sudo apt-get install -y vitis-ai-cpu

# Install Python packages
pip install --upgrade pip
pip install tensorflow opencv-python numpy scipy scikit-learn matplotlib
```

This takes about 15-20 minutes. Grab coffee ☕

**Verify installation:**
```bash
python3 -c "import tensorflow; print(tensorflow.__version__)"
# Should print version number (like 2.11.0)
```

---

## Week 2: Prepare Your Model

### Day 1-2: Get the Pre-trained Model

You already have `alzheimer_mobilenetv2_final.keras`. This is trained on 5,120 MRI images.

**The model:**
- 3.5 million parameters
- 13.8 MB (float32)
- Achieves 95% accuracy on training
- Ready for quantization

**Option A: Use the provided model** (fastest)
```bash
# This is what we recommend for first-time setup
scp alzheimer_mobilenetv2_final.keras xilinx@pynq:/home/xilinx/
```

**Option B: Train your own** (advanced)
See `alzheimer_mri_mobilenet_vitis.ipynb` for the full training code.

### Day 3-4: Quantize the Model

Quantization converts the model from 32-bit floats to 8-bit integers. This:
- Makes it 75% smaller (13.8 MB → 3.5 MB)
- Makes it run 7.7x faster
- Keeps accuracy at 93%

**Step 1: Export to SavedModel format**

Create a file `export_model.py`:

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('alzheimer_mobilenetv2_final.keras')

# Export as SavedModel (required for Vitis AI)
model.export('mobilenet_saved_model')
print("✓ Model exported successfully!")
```

Run it:
```bash
python3 export_model.py
```

**Step 2: Quantize with Vitis AI**

```bash
# Create a folder for quantization
mkdir quantized_model
cd quantized_model

# Run quantization (this is the magic command)
vai_q_tensorflow2 \
  --model /path/to/mobilenet_saved_model \
  --input_shapes 1,224,224,1 \
  --input_nodes input \
  --output_nodes predictions \
  --calib_dataset /path/to/calib_images \
  --calib_size 256 \
  --output_dir ./

# ✓ This creates quantized_model.pb
```

**What happened:**
- Model analyzed on 256 sample images
- Learned optimal int8 scale factors
- Weights converted to int8
- Result: 3.5 MB (vs original 13.8 MB)

### Day 5: Compile for FPGA

```bash
# The DPU compiler command
vai_c_tensorflow2 \
  --model quantized_model.pb \
  --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G_SG2/arch.json \
  --output_dir ./compiled \
  --net_name alzheimer_mobilenet

# ✓ This creates dpu.elf and dpu.pb
```

**Result**: FPGA machine code ready to run.

**File generated**: `dpu.elf` - This is what runs on the board's FPGA

---

## Week 3: Deploy to PYNQ Board

### Day 1-2: Transfer Files to Board

You need to copy 3 files to the PYNQ board:

```bash
# From your laptop/PC, in the folder with the model files
scp dpu.elf xilinx@pynq:/home/xilinx/
scp dpu.pb xilinx@pynq:/home/xilinx/
scp alzheimer_mobilenetv2_final.keras xilinx@pynq:/home/xilinx/
```

**Verify on board**:
```bash
# SSH into the board
ssh xilinx@pynq

# Check files are there
ls -lh *.elf *.pb *.keras
```

Should show something like:
```
-rw-r--r-- 1 xilinx xilinx 3.5M dpu.elf
-rw-r--r-- 1 xilinx xilinx 2.1M dpu.pb
-rw-r--r-- 1 xilinx xilinx 13.8M alzheimer_mobilenetv2_final.keras
```

### Day 3-4: Test Basic Inference

Create `test_inference.py`:

```python
import cv2
import numpy as np
from pynq_dpu import DpuOverlay
from vitis_ai_library import GraphRunner

# Load DPU bitstream
overlay = DpuOverlay("dpu.elf")
dpu = overlay.runner

# Test image (you need a 224x224 MRI image)
# For testing, create a random one
test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

# Preprocess (normalize and quantize)
test_image_normalized = test_image.astype(np.float32) / 255.0
test_input = test_image_normalized.reshape(1, 224, 224, 1).astype(np.int8)

# Run inference
input_data = test_input
output_data = np.zeros((1, 4), dtype=np.float32)

job_id = dpu.execute_async(input_data, output_data)
dpu.wait(job_id)

# Get result
prediction = np.argmax(output_data[0])
confidence = output_data[0][prediction]

classes = ['Non-Demented', 'Very Mild', 'Mild', 'Moderate']

print(f"✓ DPU inference works!")
print(f"  Predicted: {classes[prediction]}")
print(f"  Confidence: {confidence:.2%}")
```

Run it:
```bash
python3 test_inference.py
```

If you see the prediction, **you're in business!** ✓

### Day 5: Integration Test

Test with a real MRI image:

```bash
# Copy a real MRI image from your dataset
scp path/to/real_mri.png xilinx@pynq:/home/xilinx/

# Update test_inference.py to use the real image
# Change: test_image = cv2.imread('real_mri.png', cv2.IMREAD_GRAYSCALE)
```

Run it again. It should classify the real image!

---

## Week 4: Live Webcam Inference

### Day 1-2: Set Up Webcam

```bash
# Connect USB webcam to PYNQ board
# Verify it's detected:
ls /dev/video*

# Should show: /dev/video0 (or /dev/video1, etc)
```

### Day 3-5: Run Live Demo

Use the full webcam script from `01_COMPLETE_SETUP_GUIDE.md` (Phase 6).

Copy-paste the `webcam_inference.py` and run:

```bash
python3 webcam_inference.py
```

A window pops up with live classification! 🎉

---

## Troubleshooting Checklist

**Board doesn't respond to ping:**
- Is ethernet plugged in?
- Is power connected?
- Try: `sudo reboot` on the board

**Vitis AI not installed:**
```bash
sudo apt-get install -y vitis-ai-cpu
```

**DPU initialization fails:**
- Is `dpu.elf` in the right folder?
- Check: `ls -la dpu.elf`
- Try: `chmod +x dpu.elf`

**Inference is slow (>100ms):**
- Are you running on CPU instead of FPGA?
- Check board logs for DPU status
- Make sure you're using int8 quantized model

**Webcam shows "Failed to read":**
```bash
sudo chmod 666 /dev/video0
python3 webcam_inference.py
```

---

## Quick Reference

**Key files you'll need:**
```
alzheimer_mobilenetv2_final.keras     ← Pre-trained model
dpu.elf                               ← FPGA compiled code
dpu.pb                                ← FPGA weights
webcam_inference.py                   ← Live demo script
test_inference.py                     ← Basic test
```

**Key commands:**
```bash
# Deploy model
scp dpu.* xilinx@pynq:/home/xilinx/

# SSH to board
ssh xilinx@pynq

# Run inference
python3 test_inference.py

# Run live webcam
python3 webcam_inference.py
```

**Expected performance:**
- Inference time: 42ms
- Throughput: 23 images/sec
- Accuracy: 93%
- Power: 4W

---

## Next Steps

✅ **You've completed setup!**

Now:
1. Try different MRI images
2. Modify the code to integrate with your application
3. Experiment with the webcam positioning for best results
4. Read the other documentation for deeper understanding

Questions? Check the other markdown files:
- `02_PROJECT_ARCHITECTURE.md` - How the model works
- `03_RESULTS_TESTBENCH.md` - Performance numbers
- `01_COMPLETE_SETUP_GUIDE.md` - Detailed hardware setup (all 6 phases)

---

**You got this! 🚀**
