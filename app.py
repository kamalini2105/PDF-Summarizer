import streamlit as st
import pdfplumber
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap
import time
import io
import re

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 0.5rem 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }

    .info-card {
        background: #f8faff;
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    .summary-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
        border: 1.5px solid #c4b5fd;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #1e1b4b;
        box-shadow: 0 4px 20px rgba(167, 139, 250, 0.15);
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    .chunk-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.92rem;
        color: #374151;
        line-height: 1.7;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.15s, box-shadow 0.15s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .sidebar-section {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 999px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """Load summarization model (cached so it loads only once)."""
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
        framework="pt",
    )
    return summarizer, tokenizer


# ─── PDF Utilities ───────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> tuple[str, int]:
    """Extract raw text from an uploaded PDF, return (text, page_count)."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts), page_count


def clean_text(text: str) -> str:
    """Remove excessive whitespace and noise from extracted text."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text: str, tokenizer, max_tokens: int = 900) -> list[str]:
    """Split text into chunks that fit within the model's token limit."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        token_len = len(tokenizer.encode(sentence, truncation=False))
        if current_len + token_len > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_len = [sentence], token_len
        else:
            current_chunk.append(sentence)
            current_len += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


def summarize_chunks(
    chunks: list[str],
    summarizer,
    min_len: int,
    max_len: int,
    progress_bar,
    status_text,
) -> list[str]:
    """Summarize each chunk and return list of partial summaries."""
    summaries = []
    for i, chunk in enumerate(chunks):
        status_text.text(f"⚙️  Summarizing chunk {i + 1} of {len(chunks)}…")
        progress_bar.progress((i + 1) / len(chunks))
        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        summaries.append(result[0]["summary_text"])
        time.sleep(0.05)
    return summaries


def merge_summaries(summaries: list[str], summarizer, min_len: int, max_len: int, tokenizer) -> str:
    """If multiple chunk summaries exist, merge them into one final summary."""
    if len(summaries) == 1:
        return summaries[0]

    combined = " ".join(summaries)
    total_tokens = len(tokenizer.encode(combined))

    # If combined fits in context, do a final summarization pass
    if total_tokens <= 1000:
        result = summarizer(combined, max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]["summary_text"]

    # Otherwise join with bullet separator
    return "\n\n".join(f"• {s}" for s in summaries)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_options = {
        "BART Large CNN (Best Quality)": "facebook/bart-large-cnn",
        "T5 Small (Fastest)": "t5-small",
        "T5 Base (Balanced)": "t5-base",
        "DistilBART (Lightweight)": "sshleifer/distilbart-cnn-12-6",
    }

    selected_model_label = st.selectbox(
        "Summarization Model",
        list(model_options.keys()),
        index=0,
        help="Choose the pre-trained model for summarization.",
    )
    selected_model = model_options[selected_model_label]

    st.markdown("---")
    st.markdown("**Summary Length**")
    col1, col2 = st.columns(2)
    with col1:
        min_length = st.number_input("Min tokens", min_value=30, max_value=200, value=60, step=10)
    with col2:
        max_length = st.number_input("Max tokens", min_value=100, max_value=500, value=180, step=20)

    st.markdown("---")
    show_chunks = st.checkbox("Show chunk summaries", value=False)
    show_raw_text = st.checkbox("Show extracted text", value=False)

    st.markdown("---")
    st.markdown(
        "<small style='color:#9ca3af'>Uses HuggingFace Transformers · Runs locally · No data sent to external servers.</small>",
        unsafe_allow_html=True,
    )


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 PDF Summarizer</h1>
    <p>Upload any PDF and get an AI-powered summary in seconds</p>
</div>
""", unsafe_allow_html=True)

# ─── Upload Area ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type=["pdf"],
    help="Supports text-based PDFs up to ~50 pages for best results.",
)

if uploaded_file:
    # ── Extract text ──────────────────────────────────────────────────────
    with st.spinner("📂 Reading PDF…"):
        raw_text, page_count = extract_text_from_pdf(uploaded_file)
        clean = clean_text(raw_text)

    if not clean:
        st.error("⚠️ Could not extract text from this PDF. It may be scanned/image-based. Try an OCR tool first.")
        st.stop()

    word_count = len(clean.split())
    char_count = len(clean)

    # ── Metrics ───────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{page_count}</div><div class="metric-label">Pages</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{word_count:,}</div><div class="metric-label">Words</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{char_count:,}</div><div class="metric-label">Characters</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw Text Preview ──────────────────────────────────────────────────
    if show_raw_text:
        with st.expander("📝 Extracted Text Preview", expanded=False):
            st.text_area("", value=clean[:3000] + ("…" if len(clean) > 3000 else ""), height=200, label_visibility="collapsed")

    # ── Summarize Button ──────────────────────────────────────────────────
    if st.button("✨ Generate Summary"):
        # Load model
        load_status = st.empty()
        load_status.info(f"🔄 Loading **{selected_model_label}** — this may take a moment on first run…")
        summarizer, tokenizer = load_model(selected_model)
        load_status.empty()

        # Chunk text
        chunks = chunk_text(clean, tokenizer)

        # Progress indicators
        prog_bar = st.progress(0)
        status_txt = st.empty()

        # Summarize
        start = time.time()
        chunk_summaries = summarize_chunks(chunks, summarizer, min_length, max_length, prog_bar, status_txt)
        final_summary = merge_summaries(chunk_summaries, summarizer, min_length, max_length, tokenizer)
        elapsed = time.time() - start

        prog_bar.empty()
        status_txt.empty()

        # ── Result ────────────────────────────────────────────────────────
        st.markdown("### 🎯 Summary")
        st.markdown(f'<div class="summary-box">{final_summary}</div>', unsafe_allow_html=True)

        st.caption(f"⏱ Generated in {elapsed:.1f}s · Model: {selected_model_label} · {len(chunks)} chunk(s) processed")

        # Download button
        st.download_button(
            label="⬇️ Download Summary as .txt",
            data=final_summary,
            file_name=f"{uploaded_file.name.replace('.pdf','')}_summary.txt",
            mime="text/plain",
        )

        # ── Chunk Summaries ───────────────────────────────────────────────
        if show_chunks and len(chunk_summaries) > 1:
            st.markdown("### 🔍 Per-Chunk Summaries")
            for i, cs in enumerate(chunk_summaries):
                st.markdown(f'<div class="chunk-card"><strong>Chunk {i+1}:</strong> {cs}</div>', unsafe_allow_html=True)

else:
    # Empty state illustration
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #9ca3af;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
        <p style="font-size: 1.1rem; font-weight: 500; color: #6b7280;">Upload a PDF to get started</p>
        <p style="font-size: 0.9rem;">Supports research papers, reports, books, articles, and more</p>
    </div>
    """, unsafe_allow_html=True)
