# 🤖 answerme UI + CLI

A lightweight conversational assistant built on top of **Qwen1.5-7B-Chat**, featuring both a web-style UI and a command-line interface. This assistant can answer questions, carry a dialogue, and serve as a starting point for chatbot or AI tutor applications.

---

## ✨ Features

- Uses **Qwen1.5-7B-Chat** – a high-quality, multilingual open-source language model
- Interactive UI via **IPython widgets**
- Command-Line Interface fallback
- Efficient loading with **4-bit or 8-bit quantization** using `bitsandbytes`
- Supports GPU with `bfloat16` precision
- Maintains full **chat history** during conversations

---

## Requirements

- Python 3.8+
- GPU (recommended: T4, A100, or RTX-series)
- Jupyter Notebook or JupyterLab (for UI)

### Dependencies

Install the required packages:

```bash
pip install torch transformers bitsandbytes ipywidget

### choose interface
Choose interface (1 for UI, 2 for CLI):



