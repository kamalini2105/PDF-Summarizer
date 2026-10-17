# 📄 PDF Summarizer

A local AI-powered PDF summarizer built with **Streamlit** and **HuggingFace Transformers**. Upload any PDF and get a concise summary in seconds — no API keys required, everything runs on your machine.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multiple Models** | BART Large CNN, T5 Small/Base, DistilBART |
| **Smart Chunking** | Splits long PDFs into token-safe chunks automatically |
| **Adjustable Length** | Control min/max summary token length |
| **Download Summary** | Export as `.txt` with one click |
| **Chunk Preview** | Optionally view per-chunk summaries |
| **Raw Text View** | Inspect extracted PDF text |
| **100% Local** | No data sent to external servers |

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
cd pdf-summarizer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads PyTorch (~800 MB) and the chosen transformer model (~1.5 GB for BART Large). Subsequent runs load from cache.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🧠 Models

| Model | Size | Speed | Quality |
|---|---|---|---|
| `facebook/bart-large-cnn` | ~1.5 GB | Slow | ⭐⭐⭐⭐⭐ |
| `sshleifer/distilbart-cnn-12-6` | ~700 MB | Medium | ⭐⭐⭐⭐ |
| `t5-base` | ~250 MB | Fast | ⭐⭐⭐ |
| `t5-small` | ~60 MB | Fastest | ⭐⭐ |

---

## 📂 Project Structure

```
pdf-summarizer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🛠 How It Works

1. **Extract** — `pdfplumber` reads and extracts text from all pages.
2. **Clean** — Removes excess whitespace and formatting artifacts.
3. **Chunk** — The tokenizer splits long text into chunks within the model's context window.
4. **Summarize** — Each chunk is summarized by the chosen HuggingFace model.
5. **Merge** — Chunk summaries are combined into a single final summary.

---

## ⚙️ System Requirements

- Python 3.9+
- 4 GB RAM minimum (8 GB recommended for BART Large)
- GPU optional (CUDA auto-detected for faster inference)
- Internet required only on first run (to download models)

---

## 📝 Tips

- For best results, use text-based PDFs (not scanned/image PDFs).
- Very large PDFs (100+ pages) may take several minutes; use T5 Small for speed.
- GPU acceleration is used automatically if a CUDA-compatible GPU is available.
