# Setting Up PYNQ-ZU with Webcam

This guide walks you through getting your PYNQ-ZU board ready with a webcam for live Alzheimer classification.

## What You'll Need

**Hardware:**
- PYNQ-ZU board
- USB webcam (any standard webcam)
- 12V power supply (3A recommended)
- 16GB microSD card
- USB cable for console access
- Ethernet cable (for faster setup)

**Software:**
- PYNQ 3.0 image for ZU board
- Vitis AI toolchain (2021.1 or later)
- Python 3.8+
- OpenCV (`pip install opencv-python`)

## Step 1: Flash the SD Card

1. Download the PYNQ 3.0 image for ZU from [pynq.io](https://pynq.io)
2. Use Balena Etcher or Win32DiskImager to write to microSD card
3. Insert microSD into the board
4. Connect power and USB cable

## Step 2: Connect to Jupyter

When the board boots, it creates a Jupyter server.

**Via Ethernet (faster):**
1. Connect ethernet cable
2. Open terminal and type: `ping pynq`
3. Open browser: `http://pynq:9090`
4. Username: `xilinx`, Password: `xilinx`

**Via USB (if ethernet not available):**
1. Check Device Manager for COM port
2. Use PuTTY or similar to connect (115200 baud)
3. Run: `ifconfig` to find the IP
4. Open browser: `http://<board_ip>:9090`

## Step 3: Set Up the Board

Once in Jupyter, open a terminal from the Jupyter menu:

```bash
# Update package manager
sudo apt-get update
sudo apt-get upgrade -y

# Install OpenCV
sudo apt-get install -y python3-opencv libsm6 libxext6

# Install other dependencies
pip install --upgrade pip
pip install numpy scipy scikit-learn
```

## Step 4: Transfer the Model

Upload your quantized model to the board:

```bash
# From your local machine
scp alzheimer_mobilenetv2_final.keras xilinx@pynq:/home/xilinx/
scp dpu.bit xilinx@pynq:/home/xilinx/
```

Or use Jupyter's file upload feature directly.

## Step 5: Set Up Webcam Access

Connect your USB webcam to the board.

Check if it's detected:
```bash
ls /dev/video*
# Should show /dev/video0 or similar
```

## Step 6: Run Webcam Inference

Create a file called `webcam_inference.py`:

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

That's it! Your PYNQ-ZU is now running live Alzheimer classification with a webcam. 🎉
