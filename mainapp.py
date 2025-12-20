#  Word Cloud & Graph Analyzer (Production Ready)
#  + Standard Mode (Rich UI for small files)
#  + Streaming Mode (Low RAM for massive files)
#  + Sketch Mode (Instant loading of offline analysis)
#  + ADDED: UI Guidance, Tooltips, and Helper Text
#
import io
import re
import html
import gc
import time
import json
import math
import joblib
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Optional, Callable

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from matplotlib import font_manager
from itertools import pairwise
import openai

# --- Shared Logic Import
import text_processor as tp

# --- graph imports
import networkx as nx
import networkx.algorithms.community as nx_comm
from streamlit_agraph import agraph, Node, Edge, Config

# --- External Imports
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError: requests = None; BeautifulSoup = None

try:
    from scipy.stats import beta as beta_dist
except ImportError: beta_dist = None

try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    LatentDirichletAllocation = None; NMF = None; DictVectorizer = None; CountVectorizer = None

try:
    import openpyxl
except ImportError: openpyxl = None
try:
    import pypdf
except ImportError: pypdf = None
try:
    import pptx
except ImportError: pptx = None
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError: nltk = None; SentimentIntensityAnalyzer = None

# ------------------------------------------------
# SESSION & AUTH
# ------------------------------------------------
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False
if 'ai_response' not in st.session_state: st.session_state['ai_response'] = ""

def perform_login():
    if st.session_state.password_input == st.secrets.get("auth_password", "admin"):
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
        st.session_state['password_input'] = "" 
    else:
        st.session_state['auth_error'] = True

def logout():
    st.session_state['authenticated'] = False
    st.session_state['ai_response'] = ""

# ------------------------------------------------
# UTILITIES
# ------------------------------------------------

@st.cache_resource(show_spinner="Initializing sentiment analyzer...")
def setup_sentiment_analyzer():
    if nltk is None: return None
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

@st.cache_data(show_spinner=False)
def list_system_fonts() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for fe in font_manager.fontManager.ttflist:
        if fe.name not in mapping: mapping[fe.name] = fe.fname
    return dict(sorted(mapping.items(), key=lambda x: x[0].lower()))

def prefer_index(options: List[str], preferred: List[str]) -> int:
    for name in preferred:
        if name in options: return options.index(name)
    return 0

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

class VirtualFile:
    def __init__(self, name: str, text_content: str):
        self.name = name
        self._bytes = text_content.encode('utf-8')
    def getvalue(self) -> bytes:
        return self._bytes

def fetch_url_content(url: str) -> Optional[str]:
    if not requests or not BeautifulSoup: return None
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]): script.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        st.toast(f"Error fetching {url}: {str(e)}", icon="⚠️")
        return None

# ------------------------------------------------
# DATA PROCESSING WRAPPERS
# ------------------------------------------------

def process_rows_iter(
    rows_iter: Iterable[str],
    remove_chat: bool, remove_html: bool, unescape: bool, remove_urls: bool,
    keep_hyphens: bool, keep_apostrophes: bool,
    user_phrase_stopwords: Tuple[str, ...], user_single_stopwords: Tuple[str, ...],
    add_preps: bool, drop_integers: bool, min_word_len: int,
    compute_bigrams: bool = False, progress_cb: Optional[Callable[[int], None]] = None, update_every: int = 2_000,
) -> Dict:
    start_time = time.perf_counter()
    stopwords = set(STOPWORDS).union(user_single_stopwords)
    if add_preps: stopwords.update(tp.default_prepositions())
    trans_map = tp.build_punct_translation(keep_hyphens, keep_apostrophes)
    phrase_pattern = tp.build_phrase_pattern(list(user_phrase_stopwords))
    
    counts = Counter()
    bigram_counts = Counter() if compute_bigrams else None
    total_rows = 0

    for line in rows_iter:
        total_rows += 1
        tokens = tp.clean_and_tokenize(
            str(line) if line else "", remove_chat, remove_html, unescape, remove_urls,
            trans_map, stopwords, phrase_pattern, min_word_len, drop_integers
        )
        if tokens:
            counts.update(tokens)
            if compute_bigrams and len(tokens) > 1:
                bigram_counts.update(tuple(pairwise(tokens)))
        
        if progress_cb and (total_rows % update_every == 0): progress_cb(total_rows)

    if progress_cb: progress_cb(total_rows)
    elapsed = time.perf_counter() - start_time
    return {"counts": counts, "bigrams": bigram_counts or Counter(), "rows": total_rows, "elapsed": elapsed}


def process_large_file_stream(file_obj, chunksize, col_name, opts):
    status_text = st.empty()
    prog_bar = st.progress(0)
    word_counter = Counter()
    bigram_counter = Counter()
    total_docs = 0
    
    stop_phrases, stop_singles = tp.parse_user_stopwords(opts['stopwords'])
    stopwords = set(STOPWORDS).union(stop_singles)
    if opts['add_preps']: stopwords.update(tp.default_prepositions())
    trans_map = tp.build_punct_translation(opts['keep_hyphens'], opts['keep_apostrophes'])
    phrase_pat = tp.build_phrase_pattern(stop_phrases)

    lda = None
    vectorizer = None
    if opts['enable_topics'] and LatentDirichletAllocation:
        vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        lda = LatentDirichletAllocation(n_components=opts['n_topics'], learning_method='online', batch_size=chunksize, random_state=42)
        is_lda_init = False

    fname = file_obj.name.lower()
    
    def chunk_generator():
        file_obj.seek(0)
        if fname.endswith('.csv'):
            for chunk in pd.read_csv(file_obj, usecols=[col_name], chunksize=chunksize, on_bad_lines='skip', encoding='utf-8', engine='python'):
                yield chunk[col_name].dropna().astype(str).tolist()
        elif fname.endswith(('.xlsx', '.xlsm')):
            df = pd.read_excel(file_obj, usecols=[col_name])
            for i in range(0, len(df), chunksize):
                yield df.iloc[i:i+chunksize][col_name].dropna().astype(str).tolist()

    for i, texts in enumerate(chunk_generator()):
        clean_chunk = []
        for text in texts:
            tokens = tp.clean_and_tokenize(
                text, opts['rem_chat'], opts['rem_html'], opts['unescape'], opts['rem_urls'], 
                trans_map, stopwords, phrase_pat, opts['min_len'], opts['drop_int']
            )
            if tokens:
                word_counter.update(tokens)
                if opts['bigrams']: bigram_counter.update(tuple(pairwise(tokens)))
                if lda: clean_chunk.append(" ".join(tokens))
        
        if lda and clean_chunk:
            try:
                if not is_lda_init:
                    X = vectorizer.fit_transform(clean_chunk)
                    lda.partial_fit(X); is_lda_init = True
                else:
                    X = vectorizer.transform(clean_chunk)
                    lda.partial_fit(X)
            except: pass
        
        total_docs += len(texts)
        status_text.text(f"Processed chunk {i+1} ({total_docs:,} rows)...")
        del texts, clean_chunk
        gc.collect()

    prog_bar.progress(100)
    status_text.success(f"Streaming Complete! {total_docs:,} rows.")
    return {"word_counter": word_counter, "bigram_counter": bigram_counter, "lda_model": lda, "vectorizer": vectorizer, "total_rows": total_docs}

# ------------------------------------------------
# ML & STATS
# ------------------------------------------------

def calculate_text_stats(counts: Counter, total_rows: int) -> Dict:
    total_tokens = sum(counts.values())
    unique_tokens = len(counts)
    avg_len = sum(len(word) * count for word, count in counts.items()) / total_tokens if total_tokens else 0
    return {
        "Total Rows": total_rows, "Total Tokens": total_tokens, "Unique Vocabulary": unique_tokens,
        "Avg Word Length": round(avg_len, 2), "Lexical Diversity": round(unique_tokens / total_tokens, 4) if total_tokens else 0
    }

def extract_topics(model, vectorizer, top_n=6):
    if not model or not vectorizer: return []
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-top_n - 1:-1]
        topics.append({"id": topic_idx + 1, "words": [feature_names[i] for i in top_indices]})
    return topics

def perform_bayesian_sentiment(counts, sentiments, pos_t, neg_t, pre_pos=None, pre_neg=None):
    if not beta_dist: return None
    pos = pre_pos if pre_pos is not None else sum(counts[w] for w, s in sentiments.items() if s >= pos_t)
    neg = pre_neg if pre_neg is not None else sum(counts[w] for w, s in sentiments.items() if s <= neg_t)
    total = pos + neg
    if total < 1: return None
    a, b = 1 + pos, 1 + neg
    mean = a / (a + b)
    low, high = beta_dist.ppf([0.025, 0.975], a, b)
    x = np.linspace(0, 1, 300)
    y = beta_dist.pdf(x, a, b)
    return {"pos": pos, "neg": neg, "mean": mean, "lo": low, "hi": high, "x": x, "y": y}

# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------

def get_sentiments(_analyzer, terms):
    if not _analyzer or not terms: return {}
    return {t: _analyzer.polarity_scores(t)['compound'] for t in terms}

def create_color_func(sentiments, pos_c, neg_c, neu_c, pos_th, neg_th):
    def color_func(word, **kwargs):
        s = sentiments.get(word, 0.0)
        if s >= pos_th: return pos_c
        elif s <= neg_th: return neg_c
        return neu_c
    return color_func

def build_wordcloud(counts, max_words, w, h, bg, cmap, font, seed, color_func):
    limited = dict(counts.most_common(max_words))
    wc = WordCloud(width=w, height=h, background_color=bg, colormap=cmap, font_path=font, random_state=seed, color_func=color_func, collocations=False).generate_from_frequencies(limited)
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    plt.tight_layout()
    return fig, wc

def fig_to_png(fig):
    b = BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight")
    b.seek(0)
    return b

def generate_ai_insights(counts, bigrams, config, graph_context):
    try:
        top_u = [w for w, c in counts.most_common(100)]
        top_b = [" ".join(bg) for bg, c in bigrams.most_common(30)] if bigrams else []
        context = f"Top Unigrams: {', '.join(top_u)}\nTop Bigrams: {', '.join(top_b)}\nGraph Clusters:\n{graph_context}"
        client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'])
        response = client.chat.completions.create(
            model=config['model_name'],
            messages=[{"role": "system", "content": "Analyze these text statistics for themes."}, {"role": "user", "content": context}]
        )
        content = response.choices[0].message.content
        if response.usage:
            cost = (response.usage.prompt_tokens * config['price_in'] + response.usage.completion_tokens * config['price_out']) / 1e6
            st.session_state['total_cost'] += cost
        return content
    except Exception as e: return f"AI Error: {str(e)}"

# ------------------------------------------------
# MAIN APP UI
# ------------------------------------------------

st.set_page_config(page_title="Analytics Engine", layout="wide")
st.title("🧠 Multi-Modal Text Analytics")

# --- UI GUIDE EXPANDER ---
with st.expander("📘 Quick Guide & Glossary (Click to Expand)", expanded=False):
    st.markdown("""
    ### 📂 **Input Modes**
    1.  **Standard (Small/Medium):** The default. Use this for standard PDFs, VTTs, or CSVs/Excels with < 500k rows. 
    2.  **Streaming (Large Files):** Use this for **Massive CSVs (100MB+)**. It processes the file in chunks to prevent crashes. *Note: Requires you to type the Column Name manually.*
    3.  **Load Sketch:** Loads a `.pkl` file you previously saved. Useful for instantly reopening a massive analysis without re-waiting.

    ### 📊 **Visualizations**
    *   **Word Cloud:** Size = Frequency. Color = Sentiment (Green=Pos, Red=Neg).
    *   **Network Graph:** Shows which words appear next to each other.
    *   **Themes (LDA/NMF):** AI-detected "hidden topics" in your text.
    *   **Bayesian Sentiment:** A statistical confidence interval for how positive/negative the text really is.
    """)

analyzer = setup_sentiment_analyzer()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Config")
    
    # 1. AI Auth
    if st.session_state['authenticated']:
        st.success("AI Features Unlocked")
        ai_p = st.radio("Provider", ["xAI", "OpenAI"])
        key_name = "openai_api_key" if ai_p=="OpenAI" else "xai_api_key"
        api_key = st.secrets.get(key_name, "")
        if not api_key: api_key = st.text_input(f"{key_name}", type="password")
        ai_conf = {'api_key': api_key, 'base_url': "https://api.x.ai/v1" if ai_p=="xAI" else None, 'model_name': 'gpt-4o' if ai_p=="OpenAI" else 'grok-beta', 'price_in': 0.15, 'price_out': 0.60}
        if st.button("Logout"): logout(); st.rerun()
    else:
        with st.expander("🔐 AI Login"):
            st.text_input("Password", type="password", key="password_input", on_change=perform_login, help="Enter admin password to unlock generative AI summaries.")

    st.divider()
    
    # 2. INPUT MODE
    st.markdown("### 📂 Input Mode")
    input_mode = st.radio(
        "Select Processing Strategy", 
        ["Standard (Small/Medium Files)", "Streaming (Large Files)", "Load Sketch (.pkl)"],
        help="Choose 'Streaming' for files > 200MB to avoid running out of memory."
    )
    
    # Contextual Help Text
    if input_mode == "Standard (Small/Medium Files)":
        st.caption("✅ **Best for:** Most users. Supports PDF, VTT, JSON, and standard CSV/Excel files.")
        url_input = st.text_area("URLs (one per line)", height=60, help="App will scrape visible text from these pages.")
        manual_input = st.text_area("Manual Text", height=60)
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True, type=['csv','xlsx','pdf','txt','vtt','json'])
        
    elif input_mode == "Streaming (Large Files)":
        st.caption("🚀 **Best for:** Massive files (100MB - 1GB). Reads file in small batches. *CSV/Excel only.*")
        large_file = st.file_uploader("Upload ONE Large CSV/Excel", type=['csv', 'xlsx'])
        target_col = st.text_input("Column Name (Exact Match)", help="The exact header name of the text column in your CSV/Excel.")
        
    else:
        st.caption("💾 **Best for:** Re-opening a previously saved analysis instantly.")
        uploaded_sketch = st.file_uploader("Upload sketch.pkl", type=['pkl'])

    st.divider()

    # 3. SETTINGS
    st.markdown("### 🎨 Visuals")
    bg_color = st.color_picker("BG Color", "#ffffff")
    colormap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "tab10", "rainbow", "Blues"], index=0)
    max_words = st.slider("Max Words", 50, 3000, 1000)
    width = st.slider("Width", 600, 2400, 1200)
    height = st.slider("Height", 300, 1400, 600)
    
    st.markdown("### 🔬 Analysis")
    enable_sent = st.checkbox("Enable Sentiment", True, help="Colors words by sentiment (Green=Pos, Red=Neg). Disabling speeds up processing.")
    pos_th = st.slider("Pos Thresh", 0.0, 1.0, 0.05)
    neg_th = st.slider("Neg Thresh", -1.0, 0.0, -0.05)
    
    st.markdown("### 🧹 Cleaning")
    rem_chat = st.checkbox("No Chat Artifacts", True, help="Removes timestamps like '10:00 AM' and names like 'User:'")
    rem_html = st.checkbox("No HTML", True)
    stop_txt = st.text_area("Stopwords", "firstname.lastname", help="Comma-separated list of words/phrases to ignore.")
    user_phrase_sw, user_single_sw = tp.parse_user_stopwords(stop_txt)
    
    st.markdown("### ⚙️ Advanced")
    compute_bigrams = st.checkbox("Bigrams/Graph", True, help="Detects 2-word phrases and builds the Network Graph.")
    encoding_choice = st.selectbox("Encoding", ["auto", "latin-1"])
    topic_model_type = st.selectbox("Topic Model", ["LDA", "NMF"], help="LDA is better for long text. NMF is better for short, distinct topics.")
    n_topics = st.slider("Topics", 2, 10, 4)

# ------------------------------------------------
# MAIN PROCESSOR
# ------------------------------------------------
combined_counts = Counter()
combined_bigrams = Counter()
file_results = []
sketch_data = None
pre_calc_pos, pre_calc_neg = None, None
pre_calc_topics = None
total_processed_rows = 0

# --- PATH A: SKETCH ---
if input_mode == "Load Sketch (.pkl)" and uploaded_sketch:
    try:
        sketch_data = joblib.load(uploaded_sketch)
        combined_counts = sketch_data["word_counter"]
        combined_bigrams = sketch_data["bigram_counter"]
        pre_calc_pos = sketch_data.get("pos_count")
        pre_calc_neg = sketch_data.get("neg_count")
        total_processed_rows = sketch_data.get("total_rows", 0)
        if sketch_data.get("lda_model") and sketch_data.get("vectorizer"):
            pre_calc_topics = extract_topics(sketch_data["lda_model"], sketch_data["vectorizer"])
        st.success("Sketch Loaded!")
    except Exception as e: st.error(f"Error: {e}")

# --- PATH B: STREAMING ---
elif input_mode == "Streaming (Large Files)" and large_file and target_col:
    if st.button("Start Streaming Analysis"):
        opts = {
            'stopwords': stop_txt, 'add_preps': True, 'keep_hyphens': False, 'keep_apostrophes': False,
            'rem_chat': rem_chat, 'rem_html': rem_html, 'unescape': True, 'rem_urls': True,
            'min_len': 2, 'drop_int': True, 'bigrams': compute_bigrams, 
            'enable_topics': True, 'n_topics': n_topics
        }
        res = process_large_file_stream(large_file, 25000, target_col, opts)
        
        # Save to session state so it survives re-runs
        st.session_state['stream_res'] = res
        st.rerun()

    if 'stream_res' in st.session_state:
        res = st.session_state['stream_res']
        combined_counts = res['word_counter']
        combined_bigrams = res['bigram_counter']
        total_processed_rows = res['total_rows']
        if res['lda_model']: pre_calc_topics = extract_topics(res['lda_model'], res['vectorizer'])

# --- PATH C: STANDARD ---
elif input_mode == "Standard (Small/Medium Files)":
    all_inputs = []
    if url_input:
        for u in url_input.split('\n'):
             if u.strip(): 
                 txt = fetch_url_content(u.strip())
                 if txt: all_inputs.append(VirtualFile(f"url_{u[:10]}.txt", txt))
    if manual_input: all_inputs.append(VirtualFile("manual.txt", manual_input))
    if uploaded_files: all_inputs.extend(uploaded_files)

    if all_inputs:
        st.subheader("Processing Files")
        bar = st.progress(0)
        for i, f in enumerate(all_inputs):
            # 1. Determine Read Strategy based on file type
            fname = f.name.lower()
            bytes_data = f.getvalue()
            rows_iter = iter([])
            
            # Simple Type handling
            if fname.endswith('.vtt'): rows_iter = tp.read_rows_vtt(bytes_data, encoding_choice)
            elif fname.endswith('.pdf'): rows_iter = tp.read_rows_pdf(bytes_data)
            elif fname.endswith('.pptx'): rows_iter = tp.read_rows_pptx(bytes_data)
            elif fname.endswith('.csv'):
                # Check cols
                prev = tp.get_csv_preview(bytes_data, encoding_choice)
                if not prev.empty and len(prev.columns) > 1:
                    with st.expander(f"Select Columns for {f.name}", expanded=True):
                        cols = st.multiselect("Select Cols", prev.columns, prev.columns[0], key=f"c_{i}")
                        rows_iter = tp.iter_csv_selected_columns(bytes_data, encoding_choice, ",", True, cols)
                else: rows_iter = tp.read_rows_raw_lines(bytes_data, encoding_choice)
            else: rows_iter = tp.read_rows_raw_lines(bytes_data, encoding_choice)
            
            # 2. Process
            data = process_rows_iter(
                rows_iter, rem_chat, rem_html, True, True, False, False,
                tuple(user_phrase_sw), tuple(user_single_sw), True, True, 2, compute_bigrams
            )
            combined_counts.update(data['counts'])
            combined_bigrams.update(data['bigrams'])
            file_results.append(data)
            total_processed_rows += data['rows']
            bar.progress((i+1)/len(all_inputs))


# ------------------------------------------------
# OUTPUT DASHBOARD
# ------------------------------------------------

if combined_counts:
    # Setup Sentiment
    term_sentiments = get_sentiments(analyzer, tuple(combined_counts.keys()))
    col_func = create_color_func(term_sentiments, '#2ca02c', '#d62728', '#808080', pos_th, neg_th) if enable_sent else None

    st.divider()

    # 1. TOPIC MODELING
    st.subheader("🔍 Theme Discovery")
    topics = pre_calc_topics
    if not topics and file_results and DictVectorizer:
         # On-the-fly for standard mode
         m_type = "LDA" if "LDA" in topic_model_type else "NMF"
         with st.spinner(f"Running {m_type}..."):
             topics = tp.perform_topic_modeling([r['counts'] for r in file_results], n_topics, 6, m_type)

    if topics:
        cols = st.columns(len(topics))
        for idx, t in enumerate(topics):
            with cols[idx]:
                st.markdown(f"**Topic {t['id']}**")
                for w in t['words']: st.markdown(f"`{w}`")
    else: st.caption("No topics detected (require >1 doc or enabled setting).")

    # 2. WORD CLOUD
    st.divider()
    st.subheader("🖼️ Word Cloud")
    fig, wc = build_wordcloud(combined_counts, max_words, width, height, bg_color, colormap, None, 42, col_func)
    st.pyplot(fig, use_container_width=True)
    st.download_button("Download PNG", fig_to_png(fig), "cloud.png", "image/png")

    # 3. STATS & BAYES
    st.divider()
    text_stats = calculate_text_stats(combined_counts, total_processed_rows)
    
    st.subheader("📊 Analytics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{text_stats['Total Rows']:,}")
    c2.metric("Tokens", f"{text_stats['Total Tokens']:,}")
    c3.metric("Vocab", f"{text_stats['Unique Vocabulary']:,}")
    c4.metric("Avg Len", f"{text_stats['Avg Word Length']}")

    if enable_sent and beta_dist:
        st.caption("Bayesian Sentiment Analysis (Beta-Binomial)")
        bres = perform_bayesian_sentiment(combined_counts, term_sentiments, pos_th, neg_th, pre_calc_pos, pre_calc_neg)
        if bres:
             bc1, bc2 = st.columns([1, 2])
             bc1.success(f"95% Conf: {bres['lo']:.1%} - {bres['hi']:.1%}")
             bc1.metric("Positive Terms", f"{bres['pos']:,}")
             fig_b, ax_b = plt.subplots(figsize=(6, 2))
             ax_b.plot(bres['x'], bres['y'], color='blue', label='Posterior')
             ax_b.fill_between(bres['x'], 0, bres['y'], where=(bres['x']>bres['lo'])&(bres['x']<bres['hi']), color='green', alpha=0.3)
             ax_b.legend(); ax_b.set_title("Probability of Positive Sentiment")
             bc2.pyplot(fig_b)

    # 4. NETWORK GRAPH
    if compute_bigrams and combined_bigrams:
        st.divider()
        st.subheader("🕸️ Network Graph")
        
        # Tabs for graph details
        g_tab1, g_tab2, g_tab3 = st.tabs(["Graph Viz", "Centrality", "Data"])
        
        # Build Graph
        G = nx.Graph()
        # Filter edges for performance
        sorted_edges = combined_bigrams.most_common(1000)
        min_w = 2 if len(sorted_edges) < 100 else 5
        
        for (u, v), w in sorted_edges:
            if w >= min_w: G.add_edge(u, v, weight=w)
            
        # 4a. Viz
        with g_tab1:
            if G.number_of_nodes() > 0:
                # Physics Settings
                with st.expander("Physics"):
                    sep = st.slider("Separation", 100, 1000, 500)
                    
                # Communities
                comm_map = {}
                try:
                    for i, c in enumerate(nx_comm.greedy_modularity_communities(G)):
                         for n in c: comm_map[n] = i
                except: pass
                
                nodes, edges = [], []
                colors = ["#ff4b4b", "#4589ff", "#ffa421", "#3cdb82"]
                
                # Metrics for sizing
                deg = nx.degree_centrality(G)
                
                for n in G.nodes():
                    sz = 10 + (deg.get(n, 0) * 50)
                    c_grp = colors[comm_map.get(n, 0) % len(colors)]
                    nodes.append(Node(id=n, label=n, size=sz, color=c_grp))
                    
                for u, v, d in G.edges(data=True):
                    edges.append(Edge(source=u, target=v, color="#ddd", width=1))
                    
                agraph(nodes, edges, Config(width=900, height=600, physics=True, physicsSettings={"forceAtlas2Based": {"gravitationalConstant": -sep}}))
                
        # 4b. Centrality
        with g_tab2:
            if G.number_of_nodes() > 0:
                dc = nx.degree_centrality(G)
                bc = nx.betweenness_centrality(G, weight='weight')
                df_cent = pd.DataFrame([{"Node": n, "Degree": dc[n], "Betweenness": bc[n]} for n in G.nodes()])
                st.dataframe(df_cent.sort_values("Degree", ascending=False).head(50), use_container_width=True)

        # 4c. Data
        with g_tab3:
            st.dataframe(pd.DataFrame(sorted_edges, columns=["Bigram", "Weight"]).head(500), use_container_width=True)

    # 5. AI INSIGHTS
    if st.session_state['authenticated']:
        st.divider()
        st.subheader("🤖 AI Insights")
        if st.button("Generate Insights"):
            with st.spinner("Analyzing..."):
                g_ctx = str(comm_map) if 'comm_map' in locals() else "No graph"
                resp = generate_ai_insights(combined_counts, combined_bigrams, ai_conf, g_ctx)
                st.session_state['ai_response'] = resp
        
        if st.session_state['ai_response']:
            st.markdown(st.session_state['ai_response'])

    # 6. DOWNLOAD SKETCH (If Streaming Mode)
    if input_mode == "Streaming (Large Files)":
        st.divider()
        st.info("Save this analysis to avoid re-streaming later.")
        sketch = {
            "word_counter": combined_counts,
            "bigram_counter": combined_bigrams,
            "total_rows": total_processed_rows,
            "pos_count": pre_calc_pos,
            "neg_count": pre_calc_neg
            # (Can add LDA model if picklable, usually large though)
        }
        b = BytesIO()
        joblib.dump(sketch, b)
        st.download_button("Download .pkl Sketch", b, "analysis.pkl")

else:
    # --- EMPTY STATE / GETTING STARTED GUIDE ---
    st.info("👋 **Welcome!** Select an **Input Mode** in the sidebar to begin.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Standard")
        st.caption("For everyday files.")
        st.markdown("Drag & drop CSVs, Excel, PDFs, or Paste Text.")
    with col2:
        st.markdown("### 2. Streaming")
        st.caption("For massive datasets.")
        st.markdown("Analyzes 100MB+ files in chunks without crashing.")
    with col3:
        st.markdown("### 3. Sketch")
        st.caption("For instant replay.")
        st.markdown("Reload a previously saved analysis.")
