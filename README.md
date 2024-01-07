# Audio Compression using Huffman Coding and DCT

## Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install numpy scipy
```

## Usage

1. **Clone the repository:**
```bash
git clone https://github.com/YoussifSeliem/JPEG-compression-for-Audio
cd JPEG-compression-for-Audio
```

2. **Run the script:**
```bash
python jpeg.py
```
The script will compress and decompress the audio data, generating a reconstructed sound file.

## Overview

### File Structure

- **jpeg.py:** The main Python script containing the compression and decompression logic.
- **input.wav:** Input audio file (replace with your own audio file).
- **compressed_output:** Output directory to store compressed data.
- **reconstructed_sound.wav:** Output directory to store compressed data.

### Compression Process

1. **Sound Processing:** The input audio file is processed to extract the first channel and convert it into a 2D matrix.
2. **Block Processing:** The matrix is reshaped into blocks.
3. **DCT:** DCT is applied to each block.
4. **Quantization:** The DCT coefficients are quantized using a predefined quantization matrix.
5. **Zigzag Ordering:** The quantized coefficients are zigzag ordered.
5. **Huffman Coding:** Then Huffman coding is applied to the resulting sequence from the zigzag ordering.
6. **Output to File:** The Huffman codes and encoded data are written to an output file.

### Decompression Process

1. **Huffman Decoding:** Huffman codes are used to decode the compressed data.
2. **Inverse Zigzag Ordering:** Inverse zigzag is apllied to the sequence to get the quantized block back.
3. **Reverse Quantization:** The quantized coefficients are reconstructed using the quantization matrix.
4. **Inverse DCT:** Inverse DCT is applied to each block to obtain the reconstructed matrix.
5. **Reconstructing The Matrix:** The set blocks are recombined to form the matrix containing all blocks.
5. **Output Reconstruction:** The reconstructed matrix is converted back to a 1D array and saved to a new audio file.

## Customization

- Replace **input.wav** with your own audio file.
- Adjust parameters like the quantization matrix, block size, and file paths based on your requirements.
