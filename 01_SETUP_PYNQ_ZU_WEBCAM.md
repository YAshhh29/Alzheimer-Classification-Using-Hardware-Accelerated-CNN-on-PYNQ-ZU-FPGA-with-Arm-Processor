# 🧠 PYNQ-ZU Complete Setup & Deployment Guide

## ⚡ Hardware-Accelerated Brain MRI Analysis System

<div align="center">

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge) 
![FPGA](https://img.shields.io/badge/FPGA-Xilinx%20PYNQ--ZU-orange?style=for-the-badge) 
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-94%25-brightgreen?style=for-the-badge)
![Speedup](https://img.shields.io/badge/Speedup-7.7x-red?style=for-the-badge)

** [🏗️ Architecture](02_PROJECT_ARCHITECTURE.md) • [🧪 Results](03_RESULTS_TESTBENCH.md) • [⚙️ Implementation](04_IMPLEMENTATION_GUIDE.md)**

</div>

---

## 📋 Overview

A Comprehensive Hardware-Accelerated Edge AI System for Real-Time Alzheimer's Classification

This guide walks you through every step of setting up the PYNQ-ZU board with the Vitis AI Deep Learning Processor (DPU) for live Alzheimer's classification via webcam.

## System Architecture Overview

Your PYNQ-ZU board is a **heterogeneous computing system** - it combines:

- **ARM Cortex-A53 (CPU)**: 1.5 GHz quad-core processor on the PS (Processing System)
- **Xilinx Zynq UltraScale+ FPGA**: 117K LUTs, 1,248 DSPs in the PL (Programmable Logic)
- **Deep Learning Processor (DPU)**: Specialized co-processor for CNN operations

This means:
- ARM handles: Video capture, preprocessing, control logic
- FPGA handles: CNN inference (7.7x faster than CPU!)
- Result: Real-time classification at 23 FPS vs 3 FPS on CPU alone

## Hardware Requirements

**Essential Hardware:**
- PYNQ-ZU development board (Xilinx XCZU5EG MPSoC)
- USB webcam (UVC compliant - most standard webcams work)
- 12V DC power supply (3A minimum - important for DPU load spikes)
- 16GB Class 10 microSD card
- Ethernet cable (for Jupyter access - optional but recommended)
- Micro-USB cable (for serial console fallback)

**Why 3A power supply?** The DSP arrays in the DPU simultaneously switch state during matrix operations, creating large transient current spikes. A weak power supply will cause board resets.

**Software Stack:**
- PYNQ 3.0 image (or later) for ZU board
- Vitis AI 2.5 ecosystem
- Python 3.8+
- OpenCV (GPU-optimized image processing)
- XRT (Xilinx Runtime) drivers - manages PS/PL communication

## Phase 1: Firmware Initialization & OS Boot

The PYNQ board goes through a sophisticated multi-stage boot process. Understanding this helps if something goes wrong.

### What Happens During Boot

When you power on the board:

1. **First Stage Bootloader (FSBL)** runs from the Boot ROM
   - Initializes ARM Cortex-A53 clocks
   - Loads the base FPGA bitstream from microSD
   - Hands off to U-Boot

2. **U-Boot** runs (second stage bootloader)
   - Configures DDR4 memory
   - Loads the Linux kernel into RAM

3. **Linux Kernel** starts
   - Mounts the microSD root filesystem
   - Starts system services and Jupyter Lab server

**Total boot time:** ~30-45 seconds

### Step 1: Flash the SD Card

**Download PYNQ 3.0 for ZU:**
1. Visit [pynq.io](https://pynq.io)
2. Download the ZU variant image file (`.img`)
3. You'll get a ~4GB file

**Flash using Balena Etcher:**
1. Install [Balena Etcher](https://www.balena.io/etcher/)
2. Launch Etcher
3. Select the PYNQ image
4. Select your microSD card (⚠️ double-check the drive letter!)
5. Click "Flash"
6. Wait 5-10 minutes

**Alternative (Windows Command Line):**
```bash
# Use Win32DiskImager if Etcher doesn't work
# Or on WSL2:
sudo dd if=pynq_zu_3.0.img of=/dev/sdX bs=4M status=progress

# Replace X with your SD card letter (a, b, c, etc.)
# ⚠️ WARNING: Triple check this! dd will destroy data on wrong device
```

### Step 2: First Power-On

1. Ensure microSD card is fully inserted (click until it stops)
2. Connect 12V power supply
3. Wait 30 seconds
4. Check LEDs:
   - Green LED (LD1) should be solid
   - Blue LED (LD2) may blink (normal during boot)
   - Red LED (LD3) should NOT be on (if on, power problem)

### Step 3: Verify Network Connectivity

**Via Ethernet (recommended):**

The board creates a hostname `pynq` on your network. From your laptop:

```bash
# Windows/Mac/Linux
ping pynq

# You should see:
# Reply from 192.168.x.x: bytes=32 time=5ms TTL=64
```

If ping fails, the board might be on a different IP. Check your router's DHCP client list for "pynq" device.

**Via USB Serial (fallback):**

If Ethernet doesn't work:

1. Connect Micro-USB to board (console port on edge)
2. On Windows: Install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/)
3. Find COM port in Device Manager (usually COM3-COM5)
4. In PuTTY: 115200 baud, 8 data bits, 1 stop bit
5. Press Enter - you'll get a login prompt
6. Login: `xilinx` / Password: `xilinx`

Type `ifconfig` to see the board's IP address.

## Phase 2: Access the Jupyter IDE

Once the board is booted and connected, you'll interact with it via Jupyter Lab (a web-based Python IDE).

### Connecting to Jupyter

Open your browser and go to: **`http://pynq:9090`**

You'll see a login screen:
- **Username:** `xilinx`
- **Password:** `xilinx`

You're now in Jupyter Lab! This is where you'll run Python scripts and upload files to the board.

### Exploring Jupyter

- **Left sidebar:** File browser - see files on the board
- **Main area:** Code editor/notebook
- **Top menu:** "New" button to create terminals or notebooks

### Opening a Terminal

To run bash commands on the board:
1. Click "New" in the top-left
2. Select "Terminal"
3. You get a shell prompt on the board

**Quick terminal test:**
```bash
uname -a
# Should show: Linux pynq 5.15.x aarch64 (or similar)

python3 --version
# Should show Python 3.8 or 3.9
```

**Uploading Files:**

To copy files from your laptop to the board:
1. In Jupyter's file browser (left side), click "Upload"
2. Select your files
3. They appear in `/home/xilinx/` on the board

## Phase 3: Environment Provisioning

The board comes with a base OS, but you need to install additional libraries for video capture and deep learning.

### Step 1: Update Package Repositories

Open a terminal in Jupyter and run:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

This takes ~5-10 minutes. Grab a coffee ☕

### Step 2: Install Image Processing Libraries

```bash
# OpenCV dependencies
sudo apt-get install -y libsm6 libxext6

# Install Python OpenCV (GPU-accelerated on ARM)
sudo apt-get install -y python3-opencv

# Or use pip (usually faster):
pip install opencv-python --no-cache-dir
```

### Step 3: Install Vitis AI Deep Learning Library

This is the critical package that lets Python control the DPU (Deep Learning Processor):

```bash
# Install pynq-dpu bindings for DPU access
pip3 install pynq-dpu --no-build-isolation

# Verify installation
python3 -c "from pynq_dpu import DpuOverlay; print('✓ DPU library OK')"
```

If you see "✓ DPU library OK", you're good!

### Step 4: Install Additional Dependencies

```bash
pip install --upgrade pip

# Core ML libraries
pip install tensorflow numpy scipy scikit-learn

# Image processing and math
pip install pillow matplotlib

# Threading and async support
pip install queue  # (usually pre-installed)
```

### Verify Everything

Test that all libraries are available:

```python
# In Python shell or Jupyter notebook
import cv2
import numpy as np
import tensorflow as tf
from pynq_dpu import DpuOverlay

print("✓ All imports successful!")
print(f"  OpenCV: {cv2.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  TensorFlow: {tf.__version__}")
```

If all lines run without errors, your environment is ready!

## Phase 4: Transfer Model & Bitstream Files

Now you need to copy the compiled model and FPGA bitstream to the board.

### What Files Do You Need?

The board expects these files:

| File | Purpose | Size |
|------|---------|------|
| `dpu.bit` | FPGA bitstream (hardware programming) | ~5-10 MB |
| `dpu.hwh` | Hardware metadata (critical!) | ~1 KB |
| `dpu.xclbin` | Executable for DPU | ~2-5 MB |
| `dpu.xmodel` | Quantized MobileNetV2 model | ~3.5 MB |

The `.hwh` (Hardware Handoff) file is **crucial**. It tells the PYNQ driver where the DPU lives in memory, its clock frequency, and interrupt lines. Without it, the DPU overlay won't load.

### How to Transfer Files

**Method 1: Jupyter Upload (Easy)**

1. In Jupyter file browser (left side), navigate to `/home/xilinx/`
2. Click "Upload" button
3. Select all your `.bit`, `.hwh`, `.xclbin`, `.xmodel` files
4. Wait for upload to complete

**Method 2: SCP from Your Laptop (Fast)**

```bash
# From your laptop terminal (where the files are)
scp dpu.* xilinx@pynq:/home/xilinx/
scp alzheimer_mobilenetv2_final.keras xilinx@pynq:/home/xilinx/

# Enter password: xilinx
```

**Method 3: Direct File Sharing**

Copy files to a USB drive and plug into the board's USB ports.

### Verify Files Transferred

In the board's terminal:

```bash
ls -lh /home/xilinx/dpu.*

# You should see:
# -rw-r--r-- dpu.bit
# -rw-r--r-- dpu.hwh
# -rw-r--r-- dpu.xclbin
# -rw-r--r-- dpu.xmodel
```

If all files are there with reasonable sizes (not 0 bytes), you're good!

## Phase 5: Hardware Peripheral Verification

Before running inference, verify the DPU is loaded and the webcam is detected.

### Test DPU Initialization

Create a test file called `test_dpu.py`:

```python
# test_dpu.py
from pynq_dpu import DpuOverlay
import numpy as np

print("Loading DPU overlay...")
try:
    overlay = DpuOverlay("dpu.bit")
    dpu = overlay.runner
    print("✓ DPU loaded successfully!")
    print(f"  DPU Architecture: {dpu.get_name()}")
    print(f"  Input shape: {dpu.get_input_shape()}")
    print(f"  Output shape: {dpu.get_output_shape()}")
except Exception as e:
    print(f"✗ Failed to load DPU: {e}")
    print("  Check that dpu.bit, dpu.hwh, and dpu.xmodel are in the same directory")
```

Run it:
```bash
python3 test_dpu.py
```

If you see the architecture details, the DPU is working! 🎉

### Test Webcam Detection

The board uses the Video4Linux (V4L2) subsystem to manage USB cameras. Check if your webcam is detected:

```bash
# List video devices
ls /dev/video*

# Should show: /dev/video0 (or /dev/video1, etc)
```

If nothing shows up, your webcam might not be recognized. Try:
```bash
# Check USB devices
lsusb

# Look for something like "Bus 001 Device XXX: ID xxxx:xxxx <Vendor> <Camera>"
```

### Grant Webcam Permissions

The `xilinx` user needs permission to access the camera:

```bash
# Grant read/write access to /dev/video0
sudo chmod 666 /dev/video0

# Or permanently add xilinx to video group:
sudo usermod -aG video xilinx
# (requires reboot to take effect)
```

### Test Webcam Stream

Create `test_webcam.py`:

```python
# test_webcam.py
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Cannot open webcam")
else:
    print("✓ Webcam opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret:
        print(f"✓ Frame captured: {frame.shape}")
        # Save a test image
        cv2.imwrite('test_frame.png', frame)
        print("  Saved to test_frame.png")
    else:
        print("✗ Failed to capture frame")
    
    cap.release()
```

Run it:
```bash
python3 test_webcam.py
```

If you see "Frame captured", your webcam is working!

```python
import cv2
import numpy as np
from pynq_dpu import DpuOverlay
from vitis_ai_library import GraphRunner
import threading
import queue

# Load the DPU
overlay = DpuOverlay("dpu.bit")
dpu = overlay.runner

# Class labels
CLASS_NAMES = ['Non-Demented', 'Very Mild Dementia', 'Mild Dementia', 'Moderate Dementia']
CLASS_COLORS = [(0, 255, 0), (255, 255, 0), (255, 165, 0), (0, 0, 255)]  # Colors for display

# Input/output buffers
input_shape = (1, 224, 224, 1)  # MobileNetV2 expects 224x224 grayscale
output_shape = (1, 4)

def preprocess_frame(frame):
    """Convert frame to grayscale and resize to 224x224"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    
    # Normalize to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions: (1, 224, 224, 1)
    input_data = normalized.reshape(1, 224, 224, 1)
    
    return input_data.astype(np.int8)  # Quantized to int8

def inference_worker(frame_queue, result_queue):
    """Run inference on frames in background"""
    input_data = np.zeros(input_shape, dtype=np.int8)
    output_data = np.zeros(output_shape, dtype=np.float32)
    
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        
        input_data = preprocess_frame(frame)
        
        # Run on FPGA (async)
        job_id = dpu.execute_async(input_data, output_data)
        dpu.wait(job_id)
        
        # Get prediction
        prediction = np.argmax(output_data[0])
        confidence = output_data[0][prediction]
        
        result_queue.put({
            'class': CLASS_NAMES[prediction],
            'confidence': float(confidence),
            'color': CLASS_COLORS[prediction]
        })

# Set up queues
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

# Start inference thread
inference_thread = threading.Thread(target=inference_worker, args=(frame_queue, result_queue), daemon=True)
inference_thread.start()

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_result = None

print("Starting webcam. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam")
        break
    
    # Send frame for inference
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        pass
    
    # Get latest result if available
    try:
        latest_result = result_queue.get_nowait()
    except queue.Empty:
        pass
    
    # Draw results on frame
    if latest_result:
        text = f"{latest_result['class']} ({latest_result['confidence']:.2f})"
        color = latest_result['color']
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Alzheimer Classification - PYNQ ZU', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
```

Run it:
```bash
python3 webcam_inference.py
```

A window will pop up showing live classifications!

## Troubleshooting

**Webcam not detected:**
- Try: `sudo chmod 666 /dev/video0`
- Different board may use `/dev/video1` or higher

**Jupyter connection fails:**
- Restart board: `sudo reboot`
- Check ethernet cable connection
- Try USB connection instead

**DPU initialization error:**
- Make sure `dpu.bit` is in the same directory
- Check that Vitis AI is properly installed

**Inference is slow:**
- Make sure you're using the quantized model, not FP32
- Check that DPU is being used (not CPU fallback)

---

That's it! Your PYNQ-ZU is now running live Alzheimer classification with a webcam.
