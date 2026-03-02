"""
Autocomplete Demo — Left-to-Right & Overall Complete
IDS566 Mini Project 1 — Problem 3

Run: streamlit run app.py
"""

import streamlit as st
import re, json, random, pickle, os
import numpy as np
import nltk
from collections import defaultdict, Counter

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autocomplete Demo",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS — clean dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Main container */
.main .block-container { padding: 2rem 3rem; max-width: 1100px; }

/* Header */
.app-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2rem; padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.app-header h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.8rem; font-weight: 600;
    margin: 0; color: #f0f0f0;
}
.badge {
    display: inline-block; padding: 3px 10px;
    background: rgba(99,202,183,0.15); color: #63cab7;
    border-radius: 20px; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid rgba(99,202,183,0.3);
}

/* Mode tabs */
.mode-tab {
    padding: 10px 20px; border-radius: 8px;
    font-weight: 500; font-size: 0.9rem;
    cursor: pointer; transition: all 0.2s;
}

/* Suggestion chips */
.chip-container { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.chip {
    display: inline-block;
    padding: 6px 14px;
    background: rgba(99,202,183,0.1);
    border: 1px solid rgba(99,202,183,0.25);
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #63cab7;
    cursor: pointer;
}
.chip:hover { background: rgba(99,202,183,0.2); }

/* Completion output box */
.completion-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin: 12px 0;
    line-height: 1.8;
}
.original-text { color: #aaa; }
.generated-text {
    color: #63cab7;
    font-style: italic;
    border-bottom: 1px dashed rgba(99,202,183,0.4);
    padding-bottom: 1px;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.5rem; font-weight: 600; color: #f0f0f0; margin-top: 4px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading / training  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building language model…")
def build_model(text_path: str):
    """Load data, build vocabulary and n-gram model."""
    # ── Load ──────────────────────────────────────────────────────────────────
    with open(text_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # ── Preprocess ────────────────────────────────────────────────────────────
    def preprocess(text):
        text = re.sub(r'Human \d+:', '', text)
        text = text.lower()
        text = re.sub(r"[^a-z\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        sents = nltk.sent_tokenize(text)
        return [nltk.word_tokenize(s) for s in sents if len(s.split()) > 2]

    sentences = preprocess(raw)
    all_words  = [w for s in sentences for w in s]
    freq       = Counter(all_words)

    MIN_FREQ = 2
    vocab = ['<UNK>', '<S>', '</S>'] + sorted(w for w, c in freq.items() if c >= MIN_FREQ)
    vocab_size  = len(vocab)
    word_to_int = {w: i for i, w in enumerate(vocab)}

    def w2i(w):
        return word_to_int.get(w, 0)

    # ── N-gram counts ─────────────────────────────────────────────────────────
    bigram_counts  = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)

    for sent in sentences:
        tokens = ['<S>'] + [w if freq[w] >= MIN_FREQ else '<UNK>' for w in sent] + ['</S>']
        for i in range(len(tokens) - 1):
            bigram_counts[tokens[i]][tokens[i+1]] += 1
        for i in range(len(tokens) - 2):
            trigram_counts[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

    return {
        'sentences':     sentences,
        'vocab':         vocab,
        'vocab_size':    vocab_size,
        'word_to_int':   word_to_int,
        'bigram_counts': bigram_counts,
        'trigram_counts':trigram_counts,
        'freq':          freq,
    }


def get_text_file():
    """Locate training data — check multiple possible paths."""
    candidates = [
        'human_chat.txt',
        '/content/drive/MyDrive/miniproject/human_chat.txt',
        os.path.join(os.path.dirname(__file__), 'human_chat.txt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Autocomplete functions
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_input(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z\s']", ' ', text)
    return nltk.word_tokenize(text.strip())


def interp_prob(w1, w2, w, bigram_counts, trigram_counts, vocab_size,
                lam=0.1, alpha=0.6):
    def bigram_p(prev, nxt):
        n = bigram_counts[prev][nxt] + lam
        d = sum(bigram_counts[prev].values()) + lam * vocab_size
        return n / d

    def trigram_p(a, b, nxt):
        ctx = (a, b)
        n = trigram_counts[ctx][nxt] + lam
        d = sum(trigram_counts[ctx].values()) + lam * vocab_size
        return n / d

    return alpha * trigram_p(w1, w2, w) + (1 - alpha) * bigram_p(w2, w)


def get_top_k(context_tokens, model_data, k=8, temperature=1.0):
    vocab         = model_data['vocab']
    word_to_int   = model_data['word_to_int']
    bigram_counts = model_data['bigram_counts']
    trigram_counts= model_data['trigram_counts']
    vocab_size    = model_data['vocab_size']

    if not context_tokens:
        context_tokens = ['<s>']

    ctx = [w if w in word_to_int else '<UNK>' for w in context_tokens]
    w1  = ctx[-2] if len(ctx) >= 2 else '<S>'
    w2  = ctx[-1]

    scores = {}
    for w in vocab:
        if w in ('<S>', '<UNK>'):
            continue
        scores[w] = interp_prob(w1, w2, w, bigram_counts, trigram_counts, vocab_size)

    if temperature != 1.0:
        scores = {w: p ** (1.0 / temperature) for w, p in scores.items()}

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def left_to_right_complete(partial_text: str, model_data: dict,
                            n_suggestions=6, temperature=0.8):
    """Return top-k next word suggestions for partial_text."""
    tokens = tokenize_input(partial_text)
    if not tokens:
        return []
    candidates = get_top_k(tokens, model_data, k=n_suggestions, temperature=temperature)
    return [(w, round(p * 100, 2)) for w, p in candidates
            if w not in ('</s>', "</S>")]


def overall_complete(partial_text: str, model_data: dict,
                     max_words=25, temperature=0.9, strategy='sampling'):
    """Generate a full sentence continuation from partial_text."""
    tokens = tokenize_input(partial_text)
    if not tokens:
        return '', partial_text

    generated      = []
    current_tokens = list(tokens)

    for _ in range(max_words):
        candidates = get_top_k(current_tokens, model_data, k=20, temperature=temperature)
        if not candidates:
            break

        if strategy == 'greedy':
            next_word = candidates[0][0]
        else:
            words, probs = zip(*candidates)
            total = sum(probs)
            norm  = [p / total for p in probs]
            next_word = random.choices(words, weights=norm, k=1)[0]

        if next_word == '</S>':
            break

        generated.append(next_word)
        current_tokens.append(next_word)
        if len(current_tokens) > 8:
            current_tokens = current_tokens[-8:]

    completion = ' '.join(generated)
    full_text  = partial_text.rstrip() + (' ' if partial_text else '') + completion
    return completion, full_text


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()

    model_type = st.radio(
        "Model backend",
        ["Interpolated N-gram (recommended)", "Bigram only"],
        index=0
    )

    st.divider()
    st.markdown("**Generation parameters**")

    temperature = st.slider("Temperature", 0.3, 2.0, 0.9, 0.05,
                             help="Higher = more diverse / creative. Lower = safer / repetitive.")
    n_suggestions = st.slider("# next-word suggestions", 3, 10, 6)
    max_words = st.slider("Max completion words", 5, 50, 20)
    strategy  = st.radio("Completion strategy", ["sampling", "greedy"], index=0,
                          help="Sampling is more natural; greedy is deterministic.")

    st.divider()
    st.markdown("**Data**")
    text_path = st.text_input("Path to human_chat.txt",
                               value="human_chat.txt")

    st.divider()
    st.markdown("""
    **Design decisions:**
    - Interface: *with context*, mid-sentence
    - Left-to-right: suggests next word in real time
    - Overall: generates until end-of-sentence token
    - Hybrid interpolated trigram + bigram for reliability on small corpora
    """)


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span style="font-size:2rem">✍️</span>
    <h1>Autocomplete Demo</h1>
    <span class="badge">IDS566 · Problem 3</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────────────────────
txt_file = get_text_file() if text_path == "human_chat.txt" else text_path

if not txt_file or not os.path.exists(txt_file):
    st.error(f"⚠️  Cannot find `{text_path}`. Upload `human_chat.txt` to the same folder as `app.py`.")
    st.stop()

model_data = build_model(txt_file)

# Stats bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Sentences",  f"{len(model_data['sentences']):,}")
with c2:
    st.metric("Vocabulary", f"{model_data['vocab_size']:,}")
with c3:
    st.metric("Bigram contexts", f"{len(model_data['bigram_counts']):,}")
with c4:
    st.metric("Trigram contexts", f"{len(model_data['trigram_counts']):,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
#  Two-tab interface
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["➡️  Left-to-Right Complete", "🧩  Overall Complete"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Left-to-Right
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Type a partial sentence to get next-word suggestions")
    st.caption("The model reads your typed context and ranks the most likely next words.")

    ltr_input = st.text_area(
        "Your text so far:",
        placeholder="e.g. Hi I hope you are doing",
        height=100,
        key="ltr_input"
    )

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        ltr_go = st.button("🔍 Suggest next word", type="primary", use_container_width=True)

    if ltr_go and ltr_input.strip():
        suggestions = left_to_right_complete(
            ltr_input, model_data,
            n_suggestions=n_suggestions,
            temperature=temperature
        )

        st.markdown("**Top suggestions:**")

        # Render clickable-looking chips
        chips_html = '<div class="chip-container">'
        for word, prob in suggestions:
            chips_html += f'<span class="chip">{word} <span style="opacity:0.5;font-size:0.75em">{prob}%</span></span>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # Also show probability bar chart
        if suggestions:
            import pandas as pd
            df = pd.DataFrame(suggestions, columns=['Word', 'Probability (%)'])
            df = df.set_index('Word')
            st.bar_chart(df, height=220)

    elif ltr_go:
        st.warning("Please type something first.")

    st.divider()
    st.markdown("**Try these examples:**")
    examples_ltr = [
        "Hi I hope you are",
        "I wanted to follow up about",
        "Could you please let me know",
        "Thank you so much for your",
        "I think it would be a good",
    ]
    for ex in examples_ltr:
        if st.button(f"📝 \"{ex}\"", key=f"ex_ltr_{ex}"):
            res = left_to_right_complete(ex, model_data, n_suggestions, temperature)
            st.markdown(f"**Input:** `{ex}`")
            chips = '<div class="chip-container">' + \
                    ''.join(f'<span class="chip">{w} <span style="opacity:0.5;font-size:0.75em">{p}%</span></span>'
                            for w, p in res) + \
                    '</div>'
            st.markdown(chips, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Overall Complete
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Type a partial sentence — the model will complete it")
    st.caption("The model generates words one at a time until it predicts an end-of-sentence token or reaches the word limit.")

    oc_input = st.text_area(
        "Start of your sentence:",
        placeholder="e.g. I think it would be good",
        height=100,
        key="oc_input"
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        oc_go = st.button("🧩 Complete sentence", type="primary", use_container_width=True)
    with col_b:
        oc_multi = st.button("🎲 Show 3 variations", use_container_width=True)

    if oc_go and oc_input.strip():
        completion, full = overall_complete(
            oc_input, model_data,
            max_words=max_words,
            temperature=temperature,
            strategy=strategy
        )

        st.markdown("**Completion:**")
        html = (
            f'<div class="completion-box">'
            f'<span class="original-text">{oc_input.strip()} </span>'
            f'<span class="generated-text">{completion}</span>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

        with st.expander("📋 Copy full text"):
            st.code(full)

    elif oc_multi and oc_input.strip():
        st.markdown("**3 diverse completions (sampling):**")
        for i in range(3):
            comp, full = overall_complete(
                oc_input, model_data,
                max_words=max_words,
                temperature=temperature,
                strategy='sampling'
            )
            html = (
                f'<div class="completion-box">'
                f'<span style="color:#888;font-size:0.8em">#{i+1} &nbsp;</span>'
                f'<span class="original-text">{oc_input.strip()} </span>'
                f'<span class="generated-text">{comp}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    elif (oc_go or oc_multi):
        st.warning("Please type something first.")

    st.divider()
    st.markdown("**Try these examples:**")
    examples_oc = [
        "Hi I hope you",
        "I wanted to",
        "Could you",
        "Do you ever feel",
        "I think we should",
    ]
    for ex in examples_oc:
        if st.button(f"📝 \"{ex}\"", key=f"ex_oc_{ex}"):
            comp, full = overall_complete(ex, model_data, max_words, temperature, strategy)
            html = (
                f'<div class="completion-box">'
                f'<span class="original-text">{ex} </span>'
                f'<span class="generated-text">{comp}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
