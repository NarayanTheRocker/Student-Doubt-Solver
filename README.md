# TinyLlama + BLIP Setup Guide

This project uses:

- **TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf** (quantized Llama model)
- **BLIP Base Model** (image captioning)

Follow these steps to install everything and set up the model paths.

LLM Model: TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf.
Image Analysis Model : Blip_Base

---


## Installation

Install the required Python packages:

```bash
pip install llama-cpp-python
pip install "transformers[torch]"
pip install Pillow


your_project/
│
├── app.py
├── README.md
└── models/
     ├── tinyllama/
     │     └── TinyLlama-1.1B-Chat-v0.4-Q4_K_M.gguf
     └── blip/
           ├── processor/
           │     └── ... (tokenizer/config files)
           └── model/
                 └── ... (pytorch_model.bin and other files)

