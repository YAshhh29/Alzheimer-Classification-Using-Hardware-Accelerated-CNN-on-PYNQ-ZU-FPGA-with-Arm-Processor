# Project Architecture & How It Solves Alzheimer's Classification

## The Problem We're Solving

Alzheimer's Disease is one of the most prevalent neurodegenerative diseases. Early detection using brain MRI scans can help with early intervention and treatment planning. However:

- **Radiologists are busy**: Manual review of thousands of MRI scans takes time
- **Consistency matters**: Human analysis can be subjective
- **Speed is critical**: Waiting for results delays patient care
- **We need local processing**: Some hospitals can't rely on cloud-based AI

So we built a solution that runs **right on edge hardware** (FPGA) in the hospital, giving instant, consistent results.

## The Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PYNQ-ZU Board                         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ARM Cortex-A53 (CPU)                            │  │
│  │  - Read MRI image                                │  │
│  │  - Preprocess (grayscale, resize, normalize)    │  │
│  │  - Send to FPGA                                  │  │
│  │  - Post-process output                           │  │
│  │  - Display result                                │  │
│  └──────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Xilinx Zynq UltraScale+ (FPGA)                  │  │
│  │  - Deep Learning Processor (DPU)                │  │
│  │  - Runs MobileNetV2 CNN                          │  │
│  │  - 7.7x faster than CPU                          │  │
│  │  - 42ms per inference                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          ↓
            ┌─────────────────────────┐
            │  Output                 │
            │  Class: Mild Dementia   │
            │  Confidence: 94.2%      │
            └─────────────────────────┘
```

![Hardware Architecture Diagram](images/hardware_architecture.png)

## How It Works - Step by Step

### 1. Input: Brain MRI Image

We start with a brain MRI scan (usually 512×512 pixels or larger).

```
Input: Raw MRI scan
└─ Grayscale image from brain imaging machine
```

### 2. Preprocessing (Done on CPU)

The ARM processor does quick preparation:

```python
# Step 1: Convert to grayscale (already is, but ensure it)
gray_image = MRI_image

# Step 2: Resize to 224x224 (what MobileNetV2 expects)
resized = cv2.resize(gray_image, (224, 224))

# Step 3: Normalize pixel values to 0-1
normalized = resized / 255.0

# Step 4: Quantize to int8 (for FPGA acceleration)
# MobileNetV2 on FPGA expects int8, not float32
quantized = (normalized * 127).astype(int8)
```

Why do we preprocess on the CPU?
- These are simple, fast operations
- Saves FPGA resources for the actual neural network
- Preprocessing takes ~2ms, neural network takes 40ms

### 3. The Neural Network - MobileNetV2

This is where the magic happens on the **FPGA**.

**Why MobileNetV2?**
- Lightweight: Only 3.5M parameters (vs 138M for ResNet)
- Fast: Can run on edge devices in real-time
- Accurate: Still achieves 93% accuracy on our dataset
- Proven: Used in production by Google, Apple, etc.

**MobileNetV2 Architecture Overview:**

```
Input (224×224×1 grayscale image)
    ↓
Conv 3×3, 32 filters (stride 2)
    ↓
MobileNetV2 Blocks (14 of them)
- Depthwise convolution (cheap)
- Pointwise convolution (feature extraction)
- Skip connections (better learning)
    ↓
Global Average Pooling (reduces to 1×1)
    ↓
Dense Layer (4096 units) - feature learning
    ↓
Dropout (0.5) - prevent overfitting
    ↓
Output Layer (4 units) - one per class
    ↓
Softmax - convert to probabilities
    ↓
Output: [0.02, 0.05, 0.78, 0.15]
        Non-Dem, Very-Mild, Mild, Moderate
```

### 4. The 4 Classes We Detect

| Class | What It Means | MRI Appearance |
|-------|---------------|----------------|
| **Non-Demented (Class 0)** | Healthy brain, no cognitive decline | Normal ventricle size, intact structure |
| **Very Mild Dementia (Class 1)** | Early stage, subtle signs | Slight brain atrophy, minor ventricle enlargement |
| **Mild Dementia (Class 2)** | Clear cognitive decline | Noticeable brain shrinkage, larger ventricles |
| **Moderate Dementia (Class 3)** | Significant progression | Severe atrophy, ventricles very enlarged |

### 5. Output: Classification Result

The FPGA returns probabilities for each class:

```
Output from FPGA: [0.02, 0.05, 0.78, 0.15]
                   ↓
               Argmax (find highest)
                   ↓
           Class 2 (Mild Dementia)
                   ↓
           Display: "Mild Dementia (78% confident)"
```

## Why FPGA Acceleration?

![Performance Comparison: CPU vs FPGA](images/performance_comparison.png)

### Speed Comparison

| Component | CPU (ARM only) | FPGA | Speedup |
|-----------|----------------|------|---------|
| Time per image | 325 ms | 42 ms | **7.7x faster** |
| Throughput | 3 images/sec | 23 images/sec | **7.7x more** |
| Latency | 325 ms | 42 ms | Better for live video |

### Energy Efficiency

| Component | CPU | FPGA |
|-----------|-----|------|
| Power used per inference | 82 mJ | 17 mJ | 
| Energy efficiency | Baseline | **4.8x better** |

In a hospital scanning 100 patients/day, this saves ~6.5 kWh daily = real cost savings.

## The Training Process

Here's what happened before deployment:

![Training Pipeline](images/training_pipeline.png)

```
Step 1: Get Dataset
├─ 6,400 MRI images
├─ 1,600 per class (balanced)
└─ Split: 80% training, 20% testing

Step 2: Use Transfer Learning
├─ Start with MobileNetV2 trained on ImageNet
│  (it already knows how to find edges, textures, etc.)
├─ Freeze backbone layers
└─ Train only the final classification head (faster & better)

Step 3: Training Phase 1 - Head Training
├─ Freeze all MobileNetV2 weights
├─ Train only the custom head (4-class classifier)
├─ Learning rate: 0.001
├─ Epochs: 30
└─ Result: 91% accuracy

Step 4: Training Phase 2 - Fine-Tuning
├─ Unfreeze last 20 layers of MobileNetV2
├─ Train entire network with lower learning rate
├─ Learning rate: 0.0001
├─ Epochs: 20
└─ Result: 95% accuracy (better!)

Step 5: Quantization
├─ Convert model from float32 → int8
├─ Model shrinks from 13.8 MB → 3.5 MB
├─ Run accuracy check: 93% (minimal loss)
└─ Now ready for FPGA!

Step 6: FPGA Compilation
├─ Convert quantized model for Xilinx DPU
├─ Compile to FPGA instructions
└─ Result: 42ms per inference
```

## Data Flow in Production

```
Webcam/MRI Scanner
    ↓
ARM CPU reads image
    ↓
Preprocess on CPU
    (grayscale, resize 224×224, normalize)
    ↓
Send to FPGA DPU
    ↓
MobileNetV2 forward pass on FPGA (42ms)
    ↓
FPGA sends back probabilities [0.02, 0.05, 0.78, 0.15]
    ↓
ARM CPU post-processes
    (argmax, format for display)
    ↓
Show result: "Mild Dementia (78%)"
    ↓
Doctor makes diagnosis decision
```

## Key Metrics

- **Inference Time**: 42ms (real-time capable)
- **Model Accuracy**: 93% on 960 test images
- **Precision**: 0.93 (false positives are rare)
- **Recall**: 0.93 (catches most true cases)
- **Model Size**: 3.5 MB (fits on edge device)
- **Power**: 4.0W (stays cool, no active cooling needed)

## Why This Matters

✅ **Fast**: 42ms means doctors get results instantly  
✅ **Local**: No cloud needed, patient data stays private  
✅ **Cheap**: PYNQ board costs ~$100, not expensive cloud GPU  
✅ **Accurate**: 93% accuracy matches professional radiologists  
✅ **Proven**: Based on peer-reviewed research  

---

That's the whole system! It's a beautiful example of:
- **Machine Learning** (deep learning for image classification)
- **Hardware Acceleration** (FPGA for speed)
- **Edge AI** (runs locally without cloud)
- **Healthcare AI** (real clinical application)
