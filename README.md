# Deep JSCC-Q Web Interface

An interactive web application for demonstrating **Deep Joint Source-Channel Coding (JSCC) for Wireless Image Transmission**, specifically targeting quantized representations (Deep JSCC-Q).

This project provides a modern Flask + Keras 3 inference backend paired with a beautiful dynamic web interface. It allows users to simulate the transmission of high-resolution images over a simulated noisy wireless channel (AWGN), comparing the performance of Deep Neural Network-based joint source-channel coding against a traditional baseline (JPEG).

## Features
- **Upload Custom Images:** Test the model on any image.
- **Adjustable SNR:** Simulate different wireless channel conditions by dynamically adjusting the Signal-to-Noise Ratio (SNR).
- **Baseline Comparison:** Automatically compares the Deep JSCC reconstruction with standard JPEG transmission over the same channel constraints.
- **Metrics:** Calculates and displays PSNR and SSIM for quantitative evaluation.
- **SNR Sweep:** Automatically generate performance curves evaluating image quality across a range of SNR values.

## How to Run Locally

### Requirements
- Python 3.9+
- TensorFlow / Keras 3

### Setup & Launch
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Flask server:
```bash
python app.py
```

3. Open your web browser and navigate to `http://localhost:5000` to interact with the model.

## Credits
This implementation is based on the concepts from *Deep Joint Source-Channel Coding for Wireless Image Transmission*.
