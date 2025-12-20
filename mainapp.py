#  optimizing for public deployment + large files + graph analysis + the 'ai'
#  + BAYESIAN SENTIMENT (Beta-Binomial)
#  + SKETCH LOADING (Harvester Support)
#
import io
import re
import html
import gc
import time
import json
import math
import numpy as np
import joblib  # NEW
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Optional, Callable

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from matplotlib import font_manager
import openai

# --- Shared Logic Import
import text_processor as tp

# --- graph imports
import networkx as nx
import networkx.algorithms.community as nx_comm
from streamlit_agraph import agraph, Node, Edge, Config

# --- NEW: Web Scraping Imports
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# --- NEW: Bayesian / ML Imports
try:
    from scipy.stats import beta as beta_dist
except ImportError:
    beta_dist = None

try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction import DictVectorizer
except ImportError:
    LatentDirichletAllocation = None
    NMF = None
    DictVectorizer = None

# --- optional imports
try:
    import openpyxl
except ImportError:
    openpyxl = None
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import pptx
except ImportError:
    pptx = None
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:
    nltk = None
    SentimentIntensityAnalyzer = None

# 
# auth/session utils
# 
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False
if 'ai_response' not in st.session_state: st.session_state['ai_response'] = ""

def perform_login():
    password = st.session_state.password_input
    correct_password = st.secrets.get("auth_password", "admin")
    if password == correct_password:
        st.session_state['authenticated'] = True
        st.session_state['auth_error'] = False
        st.session_state['password_input'] = "" 
    else:
        st.session_state['auth_error'] = True

def logout():
    st.session_state['authenticated'] = False
    st.session_state['ai_response'] = ""

# 
# utilities & setup
# 

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

def parse_user_stopwords(raw: str) -> Tuple[List[str], List[str]]:
    raw = raw.replace("\n", ",").replace(".", ",")
    phrases, singles = [], []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if " " in item: phrases.append(item.lower())
        else: singles.append(item.lower())
    return phrases, singles

def build_phrase_pattern(phrases: List[str]) -> Optional[re.Pattern]:
    if not phrases: return None
    escaped = [re.escape(p) for p in phrases if p]
    if not escaped: return None
    return re.compile(rf"\b(?:{'|'.join(escaped)})\b", flags=re.IGNORECASE)

def estimate_row_count_from_bytes(file_bytes: bytes) -> int:
    if not file_bytes: return 0
    n = file_bytes.count(b"\n")
    if not file_bytes.endswith(b"\n"): n += 1
    return n

def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def make_unique_header(raw_names: List[Optional[str]]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for i, nm in enumerate(raw_names):
        name = (str(nm).strip() if nm is not None else "")
        if not name: name = f"col_{i}"
        if name in seen:
            seen[name] += 1
            unique = f"{name}__{seen[name]}"
        else:
            seen[name] = 1
            unique = name
        result.append(unique)
    return result

# --- NEW: Web Scraping Helper
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

# --- Readers (using tp) ---
# Using the helper module logic where appropriate, or keeping simpler iterators here for UI feedback

def read_rows_raw_lines(file_bytes: bytes, encoding_choice: str = "auto") -> Iterable[str]:
    def _iter_with_encoding(enc: str):
        bio = io.BytesIO(file_bytes)
        with io.TextIOWrapper(bio, encoding=enc, errors="replace", newline=None) as wrapper:
            for line in wrapper: yield line.rstrip("\r\n")
    if encoding_choice == "latin-1": yield from _iter_with_encoding("latin-1")
    else: yield from _iter_with_encoding("utf-8")

def read_rows_vtt(file_bytes: bytes, encoding_choice: str = "auto") -> Iterable[str]:
    for line in read_rows_raw_lines(file_bytes, encoding_choice):
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit(): continue
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) > 1 and len(parts[0]) < 30 and " " in parts[0]:
                yield parts[1].strip()
                continue
        yield line

def read_rows_pdf(file_bytes: bytes) -> Iterable[str]:
    if pypdf is None: return
    bio = io.BytesIO(file_bytes)
    try:
        reader = pypdf.PdfReader(bio)
        for page in reader.pages:
            text = page.extract_text()
            if text: yield text
    except Exception: yield ""

def read_rows_pptx(file_bytes: bytes) -> Iterable[str]:
    if pptx is None: return
    bio = io.BytesIO(file_bytes)
    try:
        prs = pptx.Presentation(bio)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                    if shape.text: yield shape.text
    except Exception: yield ""

def read_rows_json(file_bytes: bytes, selected_key: str = None) -> Iterable[str]:
    bio = io.BytesIO(file_bytes)
    try:
        wrapper = io.TextIOWrapper(bio, encoding="utf-8", errors="replace")
        for line in wrapper:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                if selected_key and isinstance(obj, dict): yield str(obj.get(selected_key, ""))
                elif isinstance(obj, str): yield obj
                else: yield str(obj)
            except json.JSONDecodeError:
                bio.seek(0)
                data = json.load(wrapper)
                if isinstance(data, list):
                    for item in data:
                        if selected_key and isinstance(item, dict): yield str(item.get(selected_key, ""))
                        else: yield str(item)
                break 
    except Exception: pass

def get_csv_preview(file_bytes: bytes, encoding_choice: str = "auto", delimiter: str = ",", has_header: bool = True, rows: int = 5) -> pd.DataFrame:
    enc = "latin-1" if encoding_choice == "latin-1" else "utf-8"
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio, delimiter=delimiter, header=0 if has_header else None, nrows=rows, encoding=enc, on_bad_lines='skip')
        if not has_header: df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df
    except: return pd.DataFrame()

def iter_csv_selected_columns(file_bytes: bytes, encoding_choice: str, delimiter: str, has_header: bool, selected_columns: List[str], join_with: str = " ") -> Iterable[str]:
    enc = "latin-1" if encoding_choice == "latin-1" else "utf-8"
    bio = io.BytesIO(file_bytes)
    with io.TextIOWrapper(bio, encoding=enc, errors="replace", newline="") as wrapper:
        rdr = csv.reader(wrapper, delimiter=delimiter)
        first = next(rdr, None)
        if first is None: return
        
        if has_header:
            header = make_unique_header(first)
            name_to_idx = {n: i for i, n in enumerate(header)}
        else:
            name_to_idx = {f"col_{i}": i for i in range(len(first))}
            
        idxs = [name_to_idx[n] for n in selected_columns if n in name_to_idx]
        
        # If no header, first row is data
        if not has_header:
            vals = [first[i] if i < len(first) else "" for i in idxs]
            if any(vals): yield join_with.join(str(v) for v in vals if v)

        for row in rdr:
            vals = [row[i] if i < len(row) else "" for i in idxs]
            if any(vals): yield join_with.join(str(v) for v in vals if v)

def get_excel_sheetnames(file_bytes: bytes) -> List[str]:
    if openpyxl is None: return []
    bio = io.BytesIO(file_bytes)
    wb = openpyxl.load_workbook(bio, read_only=True, data_only=True)
    sheets = list(wb.sheetnames)
    wb.close()
    return sheets

def get_excel_preview(file_bytes: bytes, sheet_name: str, has_header: bool = True, rows: int = 5) -> pd.DataFrame:
    if openpyxl is None: return pd.DataFrame()
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_excel(bio, sheet_name=sheet_name, header=0 if has_header else None, nrows=rows, engine='openpyxl')
        if not has_header: df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df
    except: return pd.DataFrame()

def iter_excel_selected_columns(file_bytes: bytes, sheet_name: str, has_header: bool, selected_columns: List[str], join_with: str = " ") -> Iterable[str]:
    if openpyxl is None: return
    bio = io.BytesIO(file_bytes)
    wb = openpyxl.load_workbook(bio, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    first = next(rows_iter, None)
    if first is None: wb.close(); return
    
    if has_header:
        header = make_unique_header(list(first))
        name_to_idx = {n: i for i, n in enumerate(header)}
    else:
        header = [f"col_{i}" for i in range(len(first))]
        name_to_idx = {n: i for i, n in enumerate(header)}
        
    idxs = [name_to_idx[n] for n in selected_columns if n in name_to_idx]
    
    if not has_header:
        vals = [first[i] if i < len(first) else "" for i in idxs]
        if any(vals): yield join_with.join("" if v is None else str(v) for v in vals if v)

    for row in rows_iter:
        vals = [row[i] if (row is not None and i < len(row)) else "" for i in idxs]
        if any(vals): yield join_with.join("" if v is None else str(v) for v in vals if v)
    wb.close()


def process_rows_iter(
    rows_iter: Iterable[str],
    remove_chat: bool, remove_html: bool, unescape: bool, remove_urls: bool,
    keep_hyphens: bool, keep_apostrophes: bool,
    user_phrase_stopwords: Tuple[str, ...], user_single_stopwords: Tuple[str, ...],
    add_preps: bool, drop_int: bool, min_len: int,
    compute_bigrams: bool = False, progress_cb: Optional[Callable[[int], None]] = None, update_every: int = 2_000,
) -> Dict:
    start_time = time.perf_counter()
    stopwords = set(STOPWORDS)
    stopwords.update(user_single_stopwords)
    if add_preps: stopwords.update(tp.default_prepositions())
    translate_map = tp.build_punct_translation(keep_hyphens, keep_apostrophes)
    phrase_pattern = build_phrase_pattern(list(user_phrase_stopwords))
    
    counts = Counter()
    bigram_counts = Counter() if compute_bigrams else None
    total_rows = 0

    for line in rows_iter:
        total_rows += 1
        tokens = tp.clean_and_tokenize(
            str(line) if line else "", remove_chat, remove_html, unescape, remove_urls,
            translate_map, stopwords, phrase_pattern, min_len, drop_int
        )
        
        if tokens:
            counts.update(tokens)
            if compute_bigrams and len(tokens) > 1:
                bigram_counts.update(tuple(bg) for bg in pairwise(tokens))
                
        if progress_cb and (total_rows % update_every == 0): progress_cb(total_rows)

    if progress_cb: progress_cb(total_rows)
    return {"counts": counts, "bigrams": bigram_counts or Counter(), "rows": total_rows, "elapsed": time.perf_counter() - start_time}

# --- Stats / ML ---

def calculate_text_stats(counts: Counter, total_rows: int) -> Dict:
    total_tokens = sum(counts.values())
    unique_tokens = len(counts)
    avg_len = sum(len(word) * count for word, count in counts.items()) / total_tokens if total_tokens else 0
    return {
        "Total Rows": total_rows,
        "Total Tokens": total_tokens,
        "Unique Vocabulary": unique_tokens,
        "Avg Word Length": round(avg_len, 2),
        "Lexical Diversity": round(unique_tokens / total_tokens, 4) if total_tokens else 0
    }

def perform_topic_modeling(file_counts: List[Counter], n_topics: int = 4, top_n_words: int = 6, model_type: str = "LDA") -> Optional[List[Dict]]:
    if not DictVectorizer: return None
    if model_type == "LDA" and not LatentDirichletAllocation: return None
    if model_type == "NMF" and not NMF: return None
    if len(file_counts) < 1: return None
    
    valid_counts = [c for c in file_counts if c and len(c) > 0]
    if not valid_counts: return None
    
    vectorizer = DictVectorizer(sparse=True)
    dtm = vectorizer.fit_transform(valid_counts)
    
    model = None
    if model_type == "LDA":
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch', max_iter=10)
    elif model_type == "NMF":
        model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    
    if not model: return None
    model.fit(dtm)
    
    return extract_topics_from_model(model, vectorizer.get_feature_names_out(), top_n_words)

def extract_topics_from_model(model, feature_names, top_n_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-top_n_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        strength = sum(topic[i] for i in top_indices)
        topics.append({"id": topic_idx + 1, "words": top_words, "strength": strength})
    return topics

def perform_bayesian_sentiment_analysis(counts: Counter, sentiments: Dict[str, float], pos_thresh: float, neg_thresh: float, pre_calc_pos=None, pre_calc_neg=None) -> Optional[Dict]:
    if not beta_dist: return None
    
    if pre_calc_pos is not None:
        pos_count = pre_calc_pos
        neg_count = pre_calc_neg
    else:
        pos_count = sum(counts[w] for w, s in sentiments.items() if s >= pos_thresh)
        neg_count = sum(counts[w] for w, s in sentiments.items() if s <= neg_thresh)
    
    total_informative = pos_count + neg_count
    if total_informative < 1: return None

    alpha_post = 1 + pos_count
    beta_post = 1 + neg_count
    mean_prob = alpha_post / (alpha_post + beta_post)
    lower_ci, upper_ci = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)
    x = np.linspace(0, 1, 300)
    y = beta_dist.pdf(x, alpha_post, beta_post)
    
    return {
        "pos_count": pos_count, "neg_count": neg_count, "total": total_informative,
        "mean_prob": mean_prob, "ci_low": lower_ci, "ci_high": upper_ci,
        "x_axis": x, "pdf_y": y
    }

# --- Visualization ---

@st.cache_data(show_spinner="Analyzing term sentiment...")
def get_sentiments(_analyzer, terms: Tuple[str, ...]) -> Dict[str, float]:
    if not _analyzer or not terms: return {}
    return {term: _analyzer.polarity_scores(term)['compound'] for term in terms}

def create_sentiment_color_func(sentiments, pos_c, neg_c, neu_c, pos_th, neg_th):
    def color_func(word, **kwargs):
        score = sentiments.get(word, 0.0)
        if score >= pos_th: return pos_c
        elif score <= neg_th: return neg_c
        return neu_c
    return color_func

def build_wordcloud_figure(counts, max_words, width, height, bg, cmap, font, seed, color_func):
    limited = dict(counts.most_common(max_words))
    wc = WordCloud(width=width, height=height, background_color=bg, colormap=cmap, font_path=font, random_state=seed, color_func=color_func, collocations=False, normalize_plurals=False).generate_from_frequencies(limited)
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    plt.tight_layout()
    return fig

def fig_to_png_bytes(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return buf

def generate_ai_insights(counts, bigrams, config, graph_context):
    try:
        top_u = [w for w, c in counts.most_common(100)]
        top_b = [" ".join(bg) for bg, c in bigrams.most_common(30)] if bigrams else []
        context = f"Unigrams: {', '.join(top_u)}\nBigrams: {', '.join(top_b)}\nClusters:\n{graph_context}"
        client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'])
        response = client.chat.completions.create(
            model=config['model_name'],
            messages=[{"role": "system", "content": "Analyze themes based on word frequencies."}, {"role": "user", "content": context}]
        )
        content = response.choices[0].message.content
        if response.usage:
            cost = (response.usage.prompt_tokens * config['price_in'] / 1e6) + (response.usage.completion_tokens * config['price_out'] / 1e6)
            st.session_state['total_cost'] += cost
        return content
    except Exception as e: return f"AI Error: {str(e)}"


# ------------------------------
# MAIN APP
# ------------------------------

st.set_page_config(page_title="Word Cloud & Graph Analytics", layout="wide")
st.title("🧠 Multi-File Word Cloud & Graph Analyzer")

with st.expander("📘 App Guide", expanded=False):
    st.markdown("""
    **Modes:**
    1. **Raw File Processing:** For standard datasets (< 500k rows).
    2. **Sketch Loader:** For massive datasets (10M+ rows). Run the `harvester.py` script offline and upload the `sketch.pkl`.
    """)

analyzer = setup_sentiment_analyzer()

# --- SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. AI Setup
    if st.session_state['authenticated']:
        st.success("AI Active")
        ai_provider = st.radio("Provider", ["xAI", "OpenAI"])
        # (Simplified for brevity, assuming existing logic)
        api_key = st.secrets.get("openai_api_key" if ai_provider=="OpenAI" else "xai_api_key", "")
        if not api_key: api_key = st.text_input("API Key", type="password")
        ai_config = {'api_key': api_key, 'base_url': None if ai_provider=="OpenAI" else "https://api.x.ai/v1", 'model_name': 'gpt-4o' if ai_provider=="OpenAI" else 'grok-4-1-fast-reasoning', 'price_in': 0.15, 'price_out': 0.60}
        if st.button("Logout"): logout(); st.rerun()
    else:
        with st.expander("Login for AI"):
            st.text_input("Password", type="password", key="password_input", on_change=perform_login)
    
    st.divider()
    
    # 2. Input Mode
    input_mode = st.radio("Input Mode", ["Raw Files / URLs", "Load Pre-computed Sketch"], index=0)
    
    # 3. File Uploaders
    if input_mode == "Load Pre-computed Sketch":
        uploaded_sketch = st.file_uploader("Upload sketch.pkl", type=["pkl"])
        uploaded_files = [] # Disable raw files
        url_input = manual_input = ""
    else:
        uploaded_sketch = None
        url_input = st.text_area("URLs", height=80)
        manual_input = st.text_area("Paste Text", height=100)
        uploaded_files = st.file_uploader("Upload Raw Files", type=["csv", "xlsx", "txt", "pdf", "json", "vtt"], accept_multiple_files=True)

    # 4. Visual Settings
    st.subheader("Visuals")
    bg_color = st.color_picker("BG Color", "#ffffff")
    colormap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "tab10", "rainbow", "Blues", "Reds"], index=0)
    max_words = st.slider("Max Words", 50, 3000, 1000)
    width = st.slider("Width", 600, 2400, 1200)
    height = st.slider("Height", 300, 1400, 600)
    
    # 5. Cleaning
    st.subheader("Cleaning")
    rem_chat = st.checkbox("No Chat Artifacts", True)
    rem_html = st.checkbox("No HTML", True)
    stop_txt = st.text_area("Stopwords", "firstname.lastname, jane doe")
    user_phrase_sw, user_single_sw = parse_user_stopwords(stop_txt)
    
    # 6. Advanced
    st.subheader("Advanced")
    compute_bigrams = st.checkbox("Bigrams / Graph", True)
    topic_model_type = st.selectbox("Topic Model", ["LDA", "NMF"])
    n_topics = st.slider("Num Topics", 2, 10, 4)
    encoding = st.selectbox("Encoding", ["auto", "latin-1"])

# -----------------------------
# PROCESSING
# -----------------------------
combined_counts = Counter()
combined_bigrams = Counter()
file_results = []
sketch_data = None
pre_calc_pos = None
pre_calc_neg = None
pre_calc_topics = None

# --- PATH A: SKETCH LOADING ---
if input_mode == "Load Pre-computed Sketch" and uploaded_sketch:
    with st.spinner("Loading Sketch..."):
        try:
            sketch_data = joblib.load(uploaded_sketch)
            combined_counts = sketch_data["word_counter"]
            combined_bigrams = sketch_data["bigram_counter"]
            
            st.success(f"Loaded Sketch! Processed {sketch_data.get('total_rows', '?'):,} rows.")
            
            # Extract Pre-computed Sentiment
            pre_calc_pos = sketch_data.get("pos_count")
            pre_calc_neg = sketch_data.get("neg_count")
            
            # Extract Pre-computed Topics
            lda_model = sketch_data.get("lda_model")
            feat_names = sketch_data.get("lda_feature_names")
            if lda_model and feat_names is not None:
                pre_calc_topics = extract_topics_from_model(lda_model, feat_names, 10)
                
        except Exception as e:
            st.error(f"Invalid Sketch File: {e}")

# --- PATH B: RAW FILES ---
elif input_mode == "Raw Files / URLs":
    all_inputs = []
    # 1. URLs
    if url_input:
        for u in url_input.split('\n'):
            if u.strip(): 
                txt = fetch_url_content(u.strip())
                if txt: all_inputs.append(VirtualFile(f"url_{u[:20]}.txt", txt))
    # 2. Manual
    if manual_input: all_inputs.append(VirtualFile("manual.txt", manual_input))
    # 3. Files
    if uploaded_files: all_inputs.extend(uploaded_files)
    
    if all_inputs:
        total_bar = st.progress(0)
        
        for i, f in enumerate(all_inputs):
            # Check for HUGE files
            if hasattr(f, "size") and f.size > 200 * 1024 * 1024:
                st.warning(f"⚠️ {f.name} is {f.size/1024/1024:.0f}MB. Consider using the 'Harvester' script offline.")
            
            # Detect Type
            fname = f.name.lower()
            bytes_data = f.getvalue()
            
            # Setup Iterator
            rows_iter = iter([])
            if fname.endswith(".vtt"): rows_iter = read_rows_vtt(bytes_data)
            elif fname.endswith(".pdf"): rows_iter = read_rows_pdf(bytes_data)
            elif fname.endswith(".pptx"): rows_iter = read_rows_pptx(bytes_data)
            elif fname.endswith(".csv"):
                df_prev = get_csv_preview(bytes_data, encoding)
                if not df_prev.empty and len(df_prev.columns) > 1:
                    # Interactive Column Selector
                    with st.expander(f"Select Columns: {f.name}", expanded=True):
                        cols = st.multiselect(f"Cols for {f.name}", df_prev.columns, df_prev.columns[0], key=f"c_{i}")
                        rows_iter = iter_csv_selected_columns(bytes_data, encoding, ",", True, cols)
                else:
                    rows_iter = read_rows_raw_lines(bytes_data, encoding)
            elif fname.endswith(".xlsx"):
                sheets = get_excel_sheetnames(bytes_data)
                if sheets:
                    rows_iter = iter_excel_selected_columns(bytes_data, sheets[0], True, get_excel_preview(bytes_data, sheets[0]).columns)
            else:
                rows_iter = read_rows_raw_lines(bytes_data, encoding)
            
            # Process
            res = process_rows_iter(
                rows_iter, rem_chat, rem_html, True, True, False, False, 
                tuple(user_phrase_sw), tuple(user_single_sw), True, True, 2, compute_bigrams
            )
            
            combined_counts.update(res["counts"])
            combined_bigrams.update(res["bigrams"])
            file_results.append(res)
            total_bar.progress((i + 1) / len(all_inputs))
            
# ---------------------------
# VISUALIZATION
# ---------------------------

if combined_counts:
    # 1. Shared Sentiment
    term_sentiments = get_sentiments(analyzer, tuple(combined_counts.keys()))
    pos_th, neg_th, pos_c, neg_c, neu_c = 0.05, -0.05, '#2ca02c', '#d62728', '#808080'
    color_func = create_sentiment_color_func(term_sentiments, pos_c, neg_c, neu_c, pos_th, neg_th)

    st.divider()
    
    # 2. Topic Modeling
    st.subheader("🔍 Theme Discovery")
    topics = None
    if pre_calc_topics:
        topics = pre_calc_topics
        st.info("Loaded pre-computed LDA topics from Sketch.")
    elif file_results:
        # On-the-fly modeling
        m_type = "LDA" if "LDA" in topic_model_type else "NMF"
        with st.spinner(f"Running {m_type}..."):
            topics = perform_topic_modeling([r['counts'] for r in file_results], n_topics, 6, m_type)
    
    if topics:
        cols = st.columns(len(topics))
        for idx, topic in enumerate(topics):
            with cols[idx]:
                st.markdown(f"**Topic {topic['id']}**")
                for w in topic['words']: st.markdown(f"`{w}`")
    else:
        st.warning("Not enough data for topic modeling.")

    # 3. Word Cloud
    st.subheader("🖼️ Word Cloud")
    fig = build_wordcloud_figure(combined_counts, max_words, width, height, bg_color, colormap, None, 42, color_func)
    st.pyplot(fig, use_container_width=True)
    st.download_button("Download PNG", fig_to_png_bytes(fig), "cloud.png", "image/png")

    # 4. Bayesian Sentiment
    st.divider()
    st.subheader("⚖️ Bayesian Sentiment")
    if beta_dist:
        bayes_res = perform_bayesian_sentiment_analysis(combined_counts, term_sentiments, pos_th, neg_th, pre_calc_pos, pre_calc_neg)
        if bayes_res:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Pos Words", f"{bayes_res['pos_count']:,}")
                st.metric("Neg Words", f"{bayes_res['neg_count']:,}")
                st.info(f"Mean Pos Rate: {bayes_res['mean_prob']:.1%}")
                st.success(f"95% CI: {bayes_res['ci_low']:.1%} — {bayes_res['ci_high']:.1%}")
            with c2:
                fig_b, ax_b = plt.subplots(figsize=(8, 3))
                ax_b.plot(bayes_res['x_axis'], bayes_res['pdf_y'], color='blue')
                ax_b.fill_between(bayes_res['x_axis'], 0, bayes_res['pdf_y'], where=(bayes_res['x_axis'] > bayes_res['ci_low']) & (bayes_res['x_axis'] < bayes_res['ci_high']), color='green', alpha=0.3)
                ax_b.set_title("Posterior Probability of Positive Sentiment")
                st.pyplot(fig_b)

    # 5. Network Graph
    if compute_bigrams and combined_bigrams:
        st.subheader("🕸️ Concept Graph")
        with st.expander("Graph Settings", expanded=False):
            min_w = st.slider("Min Edge Weight", 2, 50, 5)
            max_n = st.slider("Max Nodes", 10, 200, 60)
            
        G = nx.Graph()
        for (u, v), w in combined_bigrams.most_common(2000):
            if w >= min_w: G.add_edge(u, v, weight=w)
            if G.number_of_nodes() > max_n: break
            
        if G.number_of_nodes() > 0:
            nodes, edges = [], []
            # Communities
            comm_map = {}
            try:
                for i, c in enumerate(nx_comm.greedy_modularity_communities(G)):
                    for n in c: comm_map[n] = i
            except: pass
            
            colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"]
            for n in G.nodes():
                grp = comm_map.get(n, 0)
                nodes.append(Node(id=n, label=n, size=20, color=colors[grp % len(colors)]))
                
            for u, v, d in G.edges(data=True):
                edges.append(Edge(source=u, target=v, width=math.log(d['weight']), color="#ccc"))
            
            agraph(nodes, edges, Config(width=900, height=600, physics=True))
            
else:
    if input_mode == "Load Pre-computed Sketch" and not uploaded_sketch:
        st.info("👆 Upload a `sketch.pkl` generated by the Harvester script.")
    elif not all_inputs:
        st.info("👆 Upload files or enter text to begin.")
