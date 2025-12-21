#  THE UNSTRUCTURED DATA INTEL ENGINE
#  Architecture: Hybrid Streaming + "Data Refinery" Utility
#  Fixed: Topic Modeling Granularity Bug
#  Improved: Safety Caps, Unified Cleaning, URL Regex, Heatmap, NPMI
#
import io
import os
import re
import html
import gc
import time
import csv
import json
import math
import string
import zipfile
import tempfile
import shutil
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Iterable, Optional, Callable, Any

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from wordcloud import WordCloud, STOPWORDS
from matplotlib import font_manager
from itertools import pairwise
import openai

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

# --- PPTX Support
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

# --- PRECOMPILED PATTERNS (IMPROVED) ---
HTML_TAG_RE = re.compile(r"<[^>]+>")
CHAT_ARTIFACT_RE = re.compile(
    r":\w+:"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|yesterday) at \d{1,2}:\d{2}\b"
    r"|\b\d+\s+repl(?:y|ies)\b"
    r"|\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
    r"|\[[^\]]+\]",
    flags=re.IGNORECASE
)

# Robust URL and Email Regex
URL_EMAIL_RE = re.compile(
    r'(?:https?://|www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+[^\s]*'  # URLs
    r'|(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',          # Emails
    flags=re.IGNORECASE
)

# ---------------------------
# SKETCHING ARCHITECTURE (The "Space")
# ---------------------------

class StreamScanner:
    def __init__(self, doc_batch_size=5):
        self.global_counts = Counter()
        self.global_bigrams = Counter()
        self.total_rows_processed = 0
        
        # Topic Modeling Docs
        self.topic_docs: List[Counter] = [] 
        self.current_doc_accum = Counter()
        self.doc_accum_size = 0
        
        # Safety / Config
        self.DOC_BATCH_SIZE = doc_batch_size
        self.MAX_TOPIC_DOCS = 50_000 # Hard cap to prevent OOM
        self.limit_reached = False

    def set_batch_size(self, size: int):
        self.DOC_BATCH_SIZE = size

    def ingest_chunk_stats(self, chunk_counts: Counter, chunk_bigrams: Counter, row_count: int):
        # 1. Global Stats (Always additive, safe to grow)
        self.global_counts.update(chunk_counts)
        self.global_bigrams.update(chunk_bigrams)
        self.total_rows_processed += row_count
        
        # 2. Topic Modeling Aggregation (Needs Safety Cap)
        
        # If cap reached, we stop adding to topic_docs to save memory, 
        # but we continue processing global stats.
        if self.limit_reached:
            return

        if len(self.topic_docs) >= self.MAX_TOPIC_DOCS:
            self.limit_reached = True
            return

        # Granularity Logic
        if self.DOC_BATCH_SIZE <= 1:
            if chunk_counts:
                self.topic_docs.append(chunk_counts)
        else:
            self.current_doc_accum.update(chunk_counts)
            self.doc_accum_size += row_count
            
            if self.doc_accum_size >= self.DOC_BATCH_SIZE:
                self.topic_docs.append(self.current_doc_accum)
                self.current_doc_accum = Counter()
                self.doc_accum_size = 0

    def finalize(self):
        if not self.limit_reached and self.doc_accum_size > 0 and self.current_doc_accum:
            self.topic_docs.append(self.current_doc_accum)
            self.current_doc_accum = Counter()
            self.doc_accum_size = 0

    def to_json(self) -> str:
        serializable_bigrams = {f"{k[0]}|{k[1]}": v for k, v in self.global_bigrams.items()}
        data = {
            "total_rows": self.total_rows_processed,
            "counts": dict(self.global_counts),
            "bigrams": serializable_bigrams,
            "topic_docs": [dict(c) for c in self.topic_docs],
            "limit_reached": self.limit_reached
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str):
        try:
            data = json.loads(json_str)
            self.total_rows_processed = data.get("total_rows", 0)
            self.global_counts = Counter(data.get("counts", {}))
            raw_bigrams = data.get("bigrams", {})
            self.global_bigrams = Counter()
            for k, v in raw_bigrams.items():
                if "|" in k:
                    parts = k.split("|", 1)
                    self.global_bigrams[(parts[0], parts[1])] = v
            self.topic_docs = [Counter(d) for d in data.get("topic_docs", [])]
            self.limit_reached = data.get("limit_reached", False)
            return True
        except Exception as e:
            return False

# Initialize Session State
if 'sketch' not in st.session_state: st.session_state['sketch'] = StreamScanner()
if 'total_cost' not in st.session_state: st.session_state['total_cost'] = 0.0
if 'total_tokens' not in st.session_state: st.session_state['total_tokens'] = 0
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False
if 'auth_error' not in st.session_state: st.session_state['auth_error'] = False
if 'ai_response' not in st.session_state: st.session_state['ai_response'] = ""

def reset_sketch():
    st.session_state['sketch'] = StreamScanner()
    st.session_state['ai_response'] = ""
    gc.collect()

# 
# auth/session utils
# 

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

def prefer_index(options: List[str], preferred: List[str]) -> int:
    for name in preferred:
        if name in options: return options.index(name)
    return 0 if options else -1

@st.cache_data(show_spinner=False)
def list_system_fonts() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for fe in font_manager.fontManager.ttflist:
        if fe.name not in mapping: mapping[fe.name] = fe.fname
    return dict(sorted(mapping.items(), key=lambda x: x[0].lower()))

def build_punct_translation(keep_hyphens: bool, keep_apostrophes: bool) -> dict:
    punct = string.punctuation
    if keep_hyphens: punct = punct.replace("-", "")
    if keep_apostrophes: punct = punct.replace("'", "")
    return str.maketrans("", "", punct)

def parse_user_stopwords(raw: str) -> Tuple[List[str], List[str]]:
    raw = raw.replace("\n", ",").replace(".", ",")
    phrases, singles = [], []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if " " in item: phrases.append(item.lower())
        else: singles.append(item.lower())
    return phrases, singles

def default_prepositions() -> set:
    return {'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without'}

def build_phrase_pattern(phrases: List[str]) -> Optional[re.Pattern]:
    if not phrases: return None
    escaped = [re.escape(p) for p in phrases if p]
    if not escaped: return None
    return re.compile(rf"\b(?:{'|'.join(escaped)})\b", flags=re.IGNORECASE)

def estimate_row_count_from_bytes(file_bytes: bytes) -> int:
    # Improved to handle Windows line endings better
    if not file_bytes: return 0
    return file_bytes.count(b'\n') + 1

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
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        st.toast(f"Error fetching {url}: {str(e)}", icon="⚠️")
        return None

# 
# row readers
# 

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
    except Exception:
        yield ""

def read_rows_pptx(file_bytes: bytes) -> Iterable[str]:
    if pptx is None: return
    bio = io.BytesIO(file_bytes)
    try:
        prs = pptx.Presentation(bio)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                    if shape.text: yield shape.text
    except Exception:
        yield ""

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
                # Handle standard JSON list if not JSONL
                bio.seek(0)
                data = json.load(wrapper)
                if isinstance(data, list):
                    for item in data:
                        if selected_key and isinstance(item, dict): yield str(item.get(selected_key, ""))
                        else: yield str(item)
                elif isinstance(data, dict):
                     if selected_key: yield str(data.get(selected_key, ""))
                     else: yield str(data)
                break 
    except Exception:
        pass

# --csv/excel utils

def detect_csv_num_cols(file_bytes: bytes, encoding_choice: str = "auto", delimiter: str = ",") -> int:
    enc = "latin-1" if encoding_choice == "latin-1" else "utf-8"
    bio = io.BytesIO(file_bytes)
    try:
        with io.TextIOWrapper(bio, encoding=enc, errors="replace", newline="") as wrapper:
            rdr = csv.reader(wrapper, delimiter=delimiter)
            row = next(rdr, None)
            return len(row) if row is not None else 0
    except:
        return 0

def get_csv_columns(file_bytes: bytes, encoding_choice: str = "auto", delimiter: str = ",", has_header: bool = True) -> List[str]:
    enc = "latin-1" if encoding_choice == "latin-1" else "utf-8"
    bio = io.BytesIO(file_bytes)
    with io.TextIOWrapper(bio, encoding=enc, errors="replace", newline="") as wrapper:
        rdr = csv.reader(wrapper, delimiter=delimiter)
        first = next(rdr, None)
        if first is None: return []
        return make_unique_header(first) if has_header else [f"col_{i}" for i in range(len(first))]

def get_csv_preview(file_bytes: bytes, encoding_choice: str = "auto", delimiter: str = ",", has_header: bool = True, rows: int = 5) -> pd.DataFrame:
    enc = "latin-1" if encoding_choice == "latin-1" else "utf-8"
    bio = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(bio, delimiter=delimiter, header=0 if has_header else None, nrows=rows, encoding=enc, on_bad_lines='skip')
        if not has_header: df.columns = [f"col_{i}" for i in range(len(df.columns))]
        return df
    except:
        return pd.DataFrame()

def iter_csv_selected_columns(file_bytes: bytes, encoding_choice: str, delimiter: str, has_header: bool, selected_columns: List[str], join_with: str = " ", drop_empty: bool = True) -> Iterable[str]:
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
            vals = [first[i] if i < len(first) else "" for i in idxs]
            if drop_empty: vals = [v for v in vals if v]
            yield join_with.join(str(v) for v in vals)

        idxs = [name_to_idx[n] for n in selected_columns if n in name_to_idx]
        for row in rdr:
            vals = [row[i] if i < len(row) else "" for i in idxs]
            if drop_empty: vals = [v for v in vals if v]
            yield join_with.join(str(v) for v in vals)

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
    except:
        return pd.DataFrame()

def get_excel_columns(file_bytes: bytes, sheet_name: str, has_header: bool = True) -> List[str]:
    df = get_excel_preview(file_bytes, sheet_name, has_header, rows=1)
    if not df.empty: return list(df.columns)
    return []

def excel_estimate_rows(file_bytes: bytes, sheet_name: str, has_header: bool = True) -> int:
    if openpyxl is None: return 0
    bio = io.BytesIO(file_bytes)
    wb = openpyxl.load_workbook(bio, read_only=True, data_only=True)
    ws = wb[sheet_name]
    total = ws.max_row or 0
    wb.close()
    if has_header and total > 0: total -= 1
    return max(total, 0)

def iter_excel_selected_columns(file_bytes: bytes, sheet_name: str, has_header: bool, selected_columns: List[str], join_with: str = " ", drop_empty: bool = True) -> Iterable[str]:
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
        idxs = [name_to_idx[n] for n in selected_columns if n in name_to_idx]
    else:
        header = [f"col_{i}" for i in range(len(first))]
        name_to_idx = {n: i for i, n in enumerate(header)}
        idxs = [name_to_idx[n] for n in selected_columns if n in name_to_idx]
        vals = [first[i] if i < len(first) else "" for i in idxs]
        if drop_empty: vals = [v for v in vals if v]
        yield join_with.join("" if v is None else str(v) for v in vals)

    for row in rows_iter:
        vals = [row[i] if (row is not None and i < len(row)) else "" for i in idxs]
        if drop_empty: vals = [v for v in vals if v]
        yield join_with.join("" if v is None else str(v) for v in vals)
    wb.close()

# ---------------------------
# core processing
# 

# --- IMPROVED: Unified Cleaning Function ---
def apply_text_cleaning(
    text: str,
    remove_chat: bool, remove_html: bool, unescape: bool, remove_urls: bool,
    phrase_pattern: Optional[re.Pattern] = None
) -> str:
    if not isinstance(text, str): 
        return str(text) if text is not None else ""
        
    if remove_chat: 
        text = CHAT_ARTIFACT_RE.sub(" ", text)
    if remove_html: 
        text = HTML_TAG_RE.sub(" ", text)
    if unescape:
        try: text = html.unescape(text)
        except: pass
    
    # Improved URL removal using the robust regex
    if remove_urls:
        text = URL_EMAIL_RE.sub(" ", text)
        
    text = text.lower()
    
    # Custom Phrase Removal
    if phrase_pattern:
        text = phrase_pattern.sub(" ", text)
        
    return text.strip()

def process_chunk_iter(
    rows_iter: Iterable[str],
    remove_chat_artifacts: bool, remove_html_tags: bool, unescape_entities: bool, remove_urls: bool,
    keep_hyphens: bool, keep_apostrophes: bool,
    user_phrase_stopwords: Tuple[str, ...], user_single_stopwords: Tuple[str, ...],
    add_preps: bool, drop_integers: bool, min_word_len: int,
    compute_bigrams: bool, scanner: StreamScanner,
    progress_cb: Optional[Callable[[int], None]] = None, 
    temp_file_stats: Optional[Counter] = None
):
    stopwords = set(STOPWORDS)
    stopwords.update(user_single_stopwords)
    if add_preps: stopwords.update(default_prepositions())
    translate_map = build_punct_translation(keep_hyphens=keep_hyphens, keep_apostrophes=keep_apostrophes)
    phrase_pattern = build_phrase_pattern(list(user_phrase_stopwords))
    
    local_counts = Counter()
    local_bigrams = Counter() if compute_bigrams else Counter()
    
    is_line_by_line = scanner.DOC_BATCH_SIZE <= 1
    
    row_count = 0
    _min_len, _drop_int, _stopwords = min_word_len, drop_integers, stopwords
    _trans = translate_map

    for line in rows_iter:
        row_count += 1
        
        # --- Use Unified Cleaning ---
        text = apply_text_cleaning(
            line, 
            remove_chat_artifacts, 
            remove_html_tags, 
            unescape_entities, 
            remove_urls, 
            phrase_pattern
        )
        
        filtered_tokens_line: List[str] = []
        for t in text.split():
            # URL removal is now handled in apply_text_cleaning via regex
            t = t.translate(_trans)
            if not t or len(t) < _min_len or (_drop_int and t.isdigit()) or t in _stopwords: continue
            filtered_tokens_line.append(t)
        
        if filtered_tokens_line:
            local_counts.update(filtered_tokens_line)
            if compute_bigrams and len(filtered_tokens_line) > 1:
                local_bigrams.update(tuple(bg) for bg in pairwise(filtered_tokens_line))
            
            if is_line_by_line:
                scanner.ingest_chunk_stats(Counter(filtered_tokens_line), Counter(), 1)

        if progress_cb and (row_count % 2000 == 0): progress_cb(row_count)

    # NEW: Capture stats for the single file before merging
    if temp_file_stats is not None:
        temp_file_stats.update(local_counts)

    # Ingest AGGREGATED stats for Global Counts/Bigrams
    if not is_line_by_line:
        scanner.ingest_chunk_stats(local_counts, local_bigrams, row_count)
    else:
        # We still need to update global counts, but SKIP adding to topic_docs again
        scanner.global_counts.update(local_counts)
        scanner.global_bigrams.update(local_bigrams)
        scanner.total_rows_processed += row_count

    if progress_cb: progress_cb(row_count)
    
    del local_counts
    del local_bigrams
    gc.collect()

# --
# refinery logic (Clean & Split)
# --
def perform_refinery_job(file_obj, chunk_size, remove_chat_artifacts, remove_html_tags, unescape_entities, remove_urls, keep_hyphens, keep_apostrophes):
    """
    Reads a CSV file, applies cleaning to the text columns, saves to temp chunks, and ZIPs them.
    """
    # 1. Setup cleaning tools
    # Using the unified cleaning function, so we just need parameters
    _remove_chat = remove_chat_artifacts
    _remove_html = remove_html_tags
    _unescape = unescape_entities
    _remove_urls = remove_urls
    
    # 2. Process
    with tempfile.TemporaryDirectory() as temp_dir:
        original_name = os.path.splitext(file_obj.name)[0]
        status_container = st.status(f"⚙️ Refining {file_obj.name}...", expanded=True)
        part_num = 1
        created_files = []
        
        try:
            file_obj.seek(0)
            df_iterator = pd.read_csv(file_obj, chunksize=chunk_size, on_bad_lines='skip', dtype=str)
            
            for chunk in df_iterator:
                for col in chunk.columns:
                    chunk[col] = chunk[col].fillna("")
                    
                    # Apply cleaning lambda using the Shared Function
                    def clean_cell(text):
                        return apply_text_cleaning(text, _remove_chat, _remove_html, _unescape, _remove_urls)

                    chunk[col] = chunk[col].apply(clean_cell)
                
                # Save chunk
                new_filename = f"{original_name}_cleaned_part_{part_num}.csv"
                temp_path = os.path.join(temp_dir, new_filename)
                chunk.to_csv(temp_path, index=False)
                created_files.append(temp_path)
                status_container.write(f"✅ Processed chunk {part_num} ({len(chunk)} rows)")
                part_num += 1
            
            # Zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in created_files:
                    zip_file.write(file_path, arcname=os.path.basename(file_path))
            
            zip_buffer.seek(0)
            status_container.update(label="🎉 Refinery Job Complete!", state="complete", expanded=False)
            
            return zip_buffer
            
        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            st.error(f"Refinery Error: {str(e)}")
            return None

# --
# stats/ analytics helpers
#-

def render_interpretation_guide():
    with st.expander("🎓 Analyst's Guide: How to interpret these results", expanded=False):
        st.markdown("Use the tabs below to troubleshoot common patterns in your data.")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "😕 Graph vs. Topics Disagree", 
            "🌫️ Giant 'Blob' Graph", 
            "🔍 Topics Look Mixed", 
            "📉 Too Few Results"
        ])
        
        with tab1:
            st.markdown("""
            **Symptom:** The Network Graph shows clear, separated clusters, but the Topic Model (NMF/LDA) lumps them into one topic.
            
            **The Cause:**
            The Graph acts like a **Filter**: it hides weak connections (based on your 'Min Link Frequency' slider). 
            The Topic Model is a **Sponge**: it absorbs *every* connection, even the weak ones.
            
            **The Fix:**
            1. **Check for 'Bridges':** You likely have 1 or 2 rows of text that contain words from *both* clusters (e.g., "The **video** password **reset** is broken").
            2. **Granularity:** If 'Rows per Document' is too high, you might be accidentally merging unrelated sentences into one document. Set it to **1**.
            """)
            
        with tab2:
            st.markdown("""
            **Symptom:** The Network Graph is one giant, tangled hairball with no clear colors or clusters.
            
            **The Cause:**
            Your data is "Homogenous." Everything is related to everything else. This is common in small datasets or very specific documents (e.g., a legal contract).
            
            **The Fix:**
            1. **Increase 'Min Link Frequency':** Drag the slider up to cut the weak ties and reveal the strong skeleton of the conversation.
            2. **Repulsion:** Increase the 'Repulsion' slider in Graph Settings to physically push the nodes apart.
            """)
            
        with tab3:
            st.markdown("""
            **Symptom:** Topic 1 and Topic 2 look almost identical, or Topic 1 has all the words and Topic 2 has nonsense.
            
            **The Cause:**
            *   **Data Size:** You might not have enough data. NMF needs repetition to find patterns.
            *   **Granularity:** If 'Rows per Document' is **1**, and your sentences are very short (3-4 words), the math fails because there isn't enough context overlap.
            
            **The Fix:**
            1. **Increase Granularity:** Set 'Rows per Document' to **5** or **10**. This groups sentences together, giving the math more to work with.
            2. **Switch Model:** Try **LDA**. It is sometimes more forgiving on sparse data than NMF.
            """)

        with tab4:
            st.markdown("""
            **Symptom:** The Word Cloud is empty, or the Graph has no nodes.
            
            **The Cause:**
            Your cleaning rules are too strict.
            
            **The Fix:**
            1. **Stopwords:** Did you add too many custom stopwords?
            2. **Thresholds:** Lower the 'Top Terms Count' or the 'Min Link Frequency'.
            """)

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

# --- NEW: NPMI Calculation
def calculate_npmi(bigram_counts: Counter, unigram_counts: Counter, total_words: int, min_freq: int = 5) -> pd.DataFrame:
    results = []
    for (w1, w2), freq in bigram_counts.items():
        if freq < min_freq: continue
        
        count_w1 = unigram_counts.get(w1, 0)
        count_w2 = unigram_counts.get(w2, 0)
        
        if count_w1 == 0 or count_w2 == 0: continue
        
        prob_w1 = count_w1 / total_words
        prob_w2 = count_w2 / total_words
        prob_bigram = freq / total_words
        
        # PMI = log( p(xy) / (p(x)p(y)) )
        pmi = math.log(prob_bigram / (prob_w1 * prob_w2))
        
        # NPMI = PMI / -log(p(xy))
        npmi = pmi / -math.log(prob_bigram)
        
        results.append({"Bigram": f"{w1} {w2}", "Count": freq, "NPMI": round(npmi, 3)})
    
    return pd.DataFrame(results).sort_values("NPMI", ascending=False)

# --- Bayesian / ML Helper Functions

def perform_topic_modeling(synthetic_docs: List[Counter], n_topics: int = 4, top_n_words: int = 6, model_type: str = "LDA") -> Optional[List[Dict]]:
    if not DictVectorizer: return None
    if model_type == "LDA" and not LatentDirichletAllocation: return None
    if model_type == "NMF" and not NMF: return None
    if len(synthetic_docs) < 1: return None
    
    # 1. Vectorize
    vectorizer = DictVectorizer(sparse=True)
    dtm = vectorizer.fit_transform(synthetic_docs)
    
    # --- SAFETY CHECK FOR TINY DATA ---
    n_samples, n_features = dtm.shape
    if n_samples == 0 or n_features == 0: return None
    
    # NMF using 'nndsvd' cannot extract more topics than the smallest dimension of the data.
    # If we have 2 documents, we can't find 4 topics.
    # We automatically cap n_topics to prevent the crash.
    safe_n_topics = n_topics
    if model_type == "NMF":
        max_possible = min(n_samples, n_features)
        if safe_n_topics > max_possible:
            safe_n_topics = max_possible
    
    # If the cap reduced it to 0 (e.g. empty docs), return None
    if safe_n_topics < 1: return None
    # ----------------------------------

    # 2. Initialize Model
    model = None
    if model_type == "LDA":
        # LDA is more forgiving, but we still cap it for sanity
        safe_n_topics = min(safe_n_topics, n_samples) if n_samples < safe_n_topics else safe_n_topics
        model = LatentDirichletAllocation(n_components=safe_n_topics, random_state=42, learning_method='batch', max_iter=10)
    elif model_type == "NMF":
        model = NMF(n_components=safe_n_topics, random_state=42, init='nndsvd')
    
    if not model: return None
    
    try:
        model.fit(dtm)
    except ValueError:
        # Fallback for edge cases where nndsvd still fails
        return None
    
    # 3. Extract Topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-top_n_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        strength = sum(topic[i] for i in top_indices)
        topics.append({"id": topic_idx + 1, "words": top_words, "strength": strength})
        
    return topics

def perform_bayesian_sentiment_analysis(counts: Counter, sentiments: Dict[str, float], pos_thresh: float, neg_thresh: float) -> Optional[Dict]:
    if not beta_dist: return None
    
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

# ---------------------------
# senttiment, visualization
# 

@st.cache_data(show_spinner="Analyzing term sentiment...")
def get_sentiments(_analyzer, terms: Tuple[str, ...]) -> Dict[str, float]:
    if not _analyzer or not terms: return {}
    return {term: _analyzer.polarity_scores(term)['compound'] for term in terms}

def create_sentiment_color_func(sentiments: Dict[str, float], pos_color: str, neg_color: str, neu_color: str, pos_threshold: float, neg_threshold: float):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        score = sentiments.get(word, 0.0)
        if score >= pos_threshold: return pos_color
        elif score <= neg_threshold: return neg_color
        else: return neu_color
    return color_func

def get_sentiment_category(score: float, pos_threshold: float, neg_threshold: float) -> str:
    if score >= pos_threshold: return "Positive"
    if score <= neg_threshold: return "Negative"
    return "Neutral"

def build_wordcloud_figure_from_counts(counts: Counter, max_words: int, width: int, height: int, bg_color: str, colormap: str, font_path: Optional[str], random_state: int, color_func: Optional[Callable] = None):
    limited = dict(counts.most_common(max_words))
    wc = WordCloud(width=width, height=height, background_color=bg_color, colormap=colormap, font_path=font_path, random_state=random_state, color_func=color_func, collocations=False, normalize_plurals=False).generate_from_frequencies(limited)
    fig_w, fig_h = max(6.0, width / 100.0), max(3.0, height / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig, wc

def fig_to_png_bytes(fig: plt.Figure) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    return buf


# ai generation logic
# ---------------------------
def generate_ai_insights(counts: Counter, bigrams: Counter, config: dict, graph_context: str = ""):
    try:
        top_unigrams = [w for w, c in counts.most_common(100)]
        top_bigrams = [" ".join(bg) for bg, c in bigrams.most_common(30)] if bigrams else ["(Bigrams disabled)"]
        
        context = f"""
        Top 100 Unigrams: {', '.join(top_unigrams)}
        
        Top 30 Bigrams: {', '.join(top_bigrams)}
        
        Network Graph Clusters (detected topics):
        {graph_context}
        """
        
        system_prompt = """You are a qualitative data analyst. 
        Analyze the provided word frequency lists (extracted from a text corpus) to identify likely themes, topics, and context.
        
        Use the 'Network Graph Clusters' to specifically discuss how concepts are grouped together.
        
        Format your response with markdown headers.
        1. Likely Subject Matter
        2. Key Themes (based on Clusters)
        3. Potential Anomalies or Noise
        """
        
        client = openai.OpenAI(api_key=config['api_key'], base_url=config['base_url'])
        response = client.chat.completions.create(
            model=config['model_name'],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}]
        )
        
        content = response.choices[0].message.content
        if hasattr(response, 'usage') and response.usage:
            in_tok = response.usage.prompt_tokens
            out_tok = response.usage.completion_tokens
            cost = (in_tok * config['price_in'] / 1_000_000) + (out_tok * config['price_out'] / 1_000_000)
            st.session_state['total_tokens'] += (in_tok + out_tok)
            st.session_state['total_cost'] += cost
            
        return content
    except Exception as e:
        return f"AI Error: {str(e)}"

# -
# main app
# ------------------------------

st.set_page_config(page_title="Word Cloud & Graph Analytics", layout="wide")
st.title("🧠 Multi-File Word Cloud & Graph Analyzer")


with st.expander("📘 Comprehensive App Guide: How to use this Tool", expanded=False):
    st.markdown("""
    ### 🌟 What is this?
    This is an **Unstructured Data Intelligence Engine**. It is designed to take "dirty," raw text (from logs, surveys, transcripts, or documents) and extract mathematical structure, semantic meaning, and qualitative insights.

    ---

    ### 🛠️ Choose Your Workflow

    #### 1. The "Quick Analysis" Workflow (Small/Medium Files)
    *   **Best for:** PDFs, PowerPoints, individual Transcripts, or CSVs < 200MB.
    *   **How:** Upload files in the sidebar. 
    *   **Result:** The app processes them immediately. You get a "Quick View" Word Cloud for each file as it loads, followed by a master analysis of all files combined.

    #### 2. The "Deep Scan" Workflow (Large Datasets)
    *   **Best for:** Large CSVs (200MB - 1GB) or massive text dumps.
    *   **How:** Upload the file. Click **"Start Scan"**. 
    *   **Result:** The app switches to **Streaming Mode**. It reads the file in small chunks, extracts the statistics into a lightweight "Sketch," and immediately discards the raw text to save memory.

    #### 3. The "Enterprise" Workflow (Offline Harvesting)
    *   **Best for:** Massive corporate datasets (10M+ rows) or sensitive data that cannot leave your secure server.
    *   **How:** Use the **Offline Harvester** script (found below) to process data locally. It produces a `.json` file containing only math/statistics (no raw text) which you can upload here.

    ---

    ### 🧠 The Analytical Engines

    #### 🕸️ Network Graph & Community Detection
    *   **Concept:** Maps how words connect. Colors represent clusters of topics.
    *   **Value:** If distinct clusters appear, you have successfully separated different conversations (e.g., "Login Issues" vs. "Billing Issues").

    #### 🔥 Heatmap & Phrase Significance (NPMI)
    *   **Heatmap:** Visualizes the "neighborhood" of your top terms. See exactly how often "Battery" appears next to "Drain" vs "Charger."
    *   **NPMI (Normalized Pointwise Mutual Information):** A statistical score that separates **Meaningful Phrases** (e.g., "Artificial Intelligence") from **Random Noise** (e.g., "of the").

    #### 🔍 Bayesian Theme Discovery (Topic Modeling)
    *   **LDA:** Best for essays/assignments (assumes mixed topics).
    *   **NMF:** Best for chat logs/tickets (assumes distinct categories).
    *   *Note:* Uses a safety cap (50k docs) to prevent memory crashes on massive files.

    #### ⚖️ Bayesian Sentiment Inference
    *   **The Value:** Calculates a **Credible Interval** (e.g., "We are 95% confident the positive rate is between 55-65%") rather than a raw average, protecting you from small-sample bias.

    ---

    ### ⚡ Utility: The Data Refinery
    *   **Purpose:** Clean and split massive files that Excel can't open.
    *   **Consistency:** It now uses the **exact same** cleaning logic (Regex, URL removal, Stopwords) as the analysis engine.
    *   **Output:** A ZIP file of clean, Excel-ready CSV chunks.
    """)

analyzer = setup_sentiment_analyzer()

# --- side-bar start-
with st.sidebar:
    st.header("📂 Data Input")
    
    # 1. Main File Uploader (Top)
    st.info("Performance Tip: Streaming allows files up to ~1GB")
    uploaded_files = st.file_uploader(
        "Upload Files (csv, xlsx, json, txt, vtt, pdf, pptx)",
        type=["csv", "xlsx", "xlsm", "vtt", "txt", "json", "pdf", "pptx"],
        accept_multiple_files=True
    )
    
    clear_on_scan = st.checkbox("Clear previous data before scanning", value=True, help="If checked, scanning new files will wipe old results. If unchecked, new files are ADDED to the current analysis.")
    
    if st.button("🗑️ Reset All Data", type="primary"):
        reset_sketch()
        st.rerun()

    st.divider()

    # 2. Secondary Inputs
    with st.expander("🌐 Web, Manual & Sketch Import", expanded=False):
        sketch_upload = st.file_uploader("📂 Import Pre-computed Sketch (.json)", type=["json"], help="Skip processing by uploading a saved sketch.")
        if sketch_upload:
            if st.session_state['sketch'].load_from_json(sketch_upload.getvalue().decode('utf-8')):
                st.success("Sketch Loaded Successfully!")
            else:
                st.error("Invalid Sketch File")
        
        st.markdown("---")
        url_input = st.text_area("Enter URLs (one per line)", height=100, help="Scrape visible text.")
        manual_input = st.text_area("Paste text manually", height=150)

    st.divider()

    # 3. AI Setup
    st.header("🔐 AI Setup")
    if st.session_state['authenticated']:
        st.success("AI Features Unlocked")
        
        with st.expander("🤖 Provider Settings", expanded=True):
            ai_provider = st.radio("Provider", ["xAI (Grok)", "OpenAI (GPT-4o)"])
            
            if "OpenAI" in ai_provider:
                api_key_name = "openai_api_key"
                base_url = None 
                model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
                if "mini" in model_name:
                    price_in, price_out = 0.15, 0.60
                else:
                    price_in, price_out = 2.50, 10.00
            else:
                api_key_name = "xai_api_key"
                base_url = "https://api.x.ai/v1"
                model_options = {
                    "Grok 4.1 Fast (Reasoning) [Best Value]": "grok-4-1-fast-reasoning",
                    "Grok 4": "grok-4-0709",
                    "Grok 2 (Legacy)": "grok-2-1212"
                }
                choice = st.selectbox("Model", list(model_options.keys()))
                model_name = model_options[choice]
                
                if "fast" in model_name:
                    price_in, price_out = 0.20, 0.50
                elif "grok-4" in model_name:
                    price_in, price_out = 3.00, 15.00
                else:
                    price_in, price_out = 2.00, 10.00

            api_key = st.secrets.get(api_key_name)
            if not api_key: api_key = st.text_input(f"Enter {api_key_name}", type="password")
            
            ai_config = {
                'api_key': api_key,
                'base_url': base_url,
                'model_name': model_name,
                'price_in': price_in,
                'price_out': price_out
            }

        with st.expander("💰 Cost Estimator", expanded=False):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Tokens:**\n{st.session_state['total_tokens']:,}")
            c2.markdown(f"**Cost:**\n`${st.session_state['total_cost']:.5f}`")
            if st.button("Reset Cost"):
                st.session_state['total_cost'] = 0.0
                st.session_state['total_tokens'] = 0
                st.rerun()
        
        if st.button("Logout"): logout(); st.rerun()
    else:
        with st.expander("Unlock AI Features", expanded=True):
            st.text_input("Password", type="password", key="password_input", on_change=perform_login)
            if st.session_state['auth_error']: st.error("Incorrect password.")

    st.divider()

    st.markdown("### 🎨 appearance")
    bg_color = st.color_picker("background color", value="#ffffff")
    colormap = st.selectbox("colormap", options=["viridis", "plasma", "inferno", "magma", "cividis", "tab10", "tab20", "Dark2", "Set3", "rainbow", "cubehelix", "prism", "Blues", "Greens", "Oranges", "Reds", "Purples", "Greys"], index=0)
    max_words = st.slider("max words in word cloud", 50, 3000, 1000, 50)
    width = st.slider("image width (px)", 600, 2400, 1200, 100)
    height = st.slider("image height (px)", 300, 1400, 600, 50)
    random_state = st.number_input("random seed", 0, value=42, step=1)

    st.markdown("### 🔬 sentiment analysis")
    enable_sentiment = st.checkbox("enable sentiment analysis", value=False)
    if enable_sentiment and analyzer is None:
        st.error("NLTK not found.")
        enable_sentiment = False
    pos_threshold, neg_threshold, pos_color, neu_color, neg_color = 0.05, -0.05, '#2ca02c', '#808080', '#d62728'
    if enable_sentiment:
        c1, c2 = st.columns(2)
        with c1: pos_threshold = st.slider("pos threshold", 0.0, 1.0, 0.05, 0.01)
        with c2: neg_threshold = st.slider("neg threshold", -1.0, 0.0, -0.05, 0.01)
        c1, c2, c3 = st.columns(3)
        with c1: pos_color = st.color_picker("pos color", value=pos_color)
        with c2: neu_color = st.color_picker("neu color", value=neu_color)
        with c3: neg_color = st.color_picker("neg color", value=neg_color)

    st.markdown("### 🧹 cleaning")
    remove_chat_artifacts = st.checkbox("remove chat artifacts", value=True)
    remove_html_tags = st.checkbox("strip html tags", value=True)
    unescape_entities = st.checkbox("unescape html entities", value=True)
    remove_urls = st.checkbox("remove urls", value=True)
    keep_hyphens = st.checkbox("keep hyphens", value=False)
    keep_apostrophes = st.checkbox("keep apostrophes", value=False)

    st.markdown("### 🛑 stopwords")
    user_input = st.text_area("custom stopwords (comma-separated)", value="firstname.lastname, jane doe")
    user_phrase_stopwords, user_single_stopwords = parse_user_stopwords(user_input)
    add_preps = st.checkbox("remove prepositions", value=True)
    drop_integers = st.checkbox("remove integers", value=True)
    min_word_len = st.slider("min word length", 1, 10, 2)

    st.markdown("### 📊 tables & font")
    top_n = st.number_input("top terms count", 5, 10000, 20)
    font_map, font_names = list_system_fonts(), list(list_system_fonts().keys())
    preferred_defaults = ["cmtt10", "cmr10", "Arial", "DejaVu Sans", "Helvetica", "Verdana"]
    default_font_index = prefer_index(font_names, preferred_defaults)
    combined_font_name = st.selectbox("font for combined cloud", font_names or ["(default)"], max(default_font_index, 0))
    combined_font_path = font_map.get(combined_font_name) if font_names else None
    
    with st.expander("⚙️ performance options", expanded=True):
        encoding_choice = st.selectbox("file encoding", ["auto (utf-8)", "latin-1"])
        chunksize = st.number_input("csv chunk size", 1_000, 100_000, 10_000, 1_000)
        compute_bigrams = st.checkbox("compute bigrams / graph", value=True)
        # New Settings for Topic Modeling
        st.markdown("**Topic Modeling (LDA/NMF)**")
        topic_model_type = st.selectbox("Model Type", ["LDA (Probabilistic)", "NMF (Distinct)"], index=0, help="LDA is better for long text/essays. NMF is better for short logs/chats.")
        n_topics_val = st.slider("Number of Topics", 2, 10, 4, help="How many hidden themes to search for.")
        
        # NEW: Granularity Control (FIXED OPTIONS)
        st.markdown("**Processing Granularity**")
        doc_granularity = st.select_slider(
            "Rows per Document",
            options=[1, 5, 10, 100, 500],
            value=5,
            help="1 = Every line is a document (Best for small files). 500 = Groups rows (Best for massive files)."
        )
        
        # --- FIXED: GRANULARITY CONSISTENCY CHECK ---
        # Initialize
        if 'last_granularity' not in st.session_state:
            st.session_state['last_granularity'] = doc_granularity

        # Check for change
        if st.session_state['last_granularity'] != doc_granularity:
            # Only reset if we actually have data to lose
            if st.session_state['sketch'].total_rows_processed > 0:
                st.warning(f"Granularity changed ({st.session_state['last_granularity']} → {doc_granularity}). Previous data reset to ensure consistent Topic Modeling.")
                reset_sketch()
            st.session_state['last_granularity'] = doc_granularity

        # Ensure sketch always has current setting (Immediately, not just on scan)
        st.session_state['sketch'].set_batch_size(doc_granularity)
        # ----------------------------------------------

# -----------------------------
# NEW: DATA REFINERY (UTILITY SECTION)
# --------------------------
with st.expander("🛠️ Data Refinery: Split & Clean Massive CSVs", expanded=False):
    st.markdown("""
    **The Refinery** is a utility for data engineering. It creates CLEANED copies of your files.
    1. Upload a CSV (even a large one).
    2. The app reads it in chunks, applies your cleaning settings (removing HTML, timestamps, etc.).
    3. It splits the file into smaller parts and returns a ZIP file.
    """)
    refinery_file = st.file_uploader("Upload CSV to Refine:", type=['csv'])
    r_rows = st.number_input("Rows per split file:", 1000, 500000, 50000)
    
    if refinery_file and st.button("🚀 Start Refinery Job"):
        zip_data = perform_refinery_job(
            refinery_file, r_rows, 
            remove_chat_artifacts, remove_html_tags, unescape_entities, remove_urls, keep_hyphens, keep_apostrophes
        )
        if zip_data:
            st.download_button(
                label="📥 Download Cleaned & Split ZIP",
                data=zip_data,
                file_name=f"{os.path.splitext(refinery_file.name)[0]}_refined.zip",
                mime="application/zip",
                type="primary"
            )

# -----------------------------
# main processing loop (SCANNING)
# --------------------------

# --- NEW: ENTERPRISE GUIDE (For offline harvesting) ---
with st.expander("🚀 Enterprise Workflow: Processing 10M+ Rows Locally", expanded=False):
    st.markdown("""
    For massive datasets that are too large to upload, use this **Offline Harvester**.
    
    **Instructions:**
    1.  Copy the code below into a file named `harvest.py` on your server/laptop.
    2.  Run it against your large CSV: `python harvest.py my_huge_file.csv`
    3.  It will produce `my_huge_file_sketch.json`.
    4.  Upload that JSON file here using the **"Import Pre-computed Sketch"** button in the sidebar.
    """)
    st.code("""
import sys
import csv
import json
import collections
import re
from itertools import pairwise

# Minimal Harvester Script
def simple_tokenize(text):
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text) # strip html
    text = re.sub(r'[^a-z0-9\\s]', '', text) # punct
    return [w for w in text.split() if len(w) > 2 and not w.isdigit()]

def harvest(filename):
    print(f"Harvesting {filename}...")
    
    global_counts = collections.Counter()
    global_bigrams = collections.Counter()
    topic_docs = []
    current_doc = collections.Counter()
    doc_size = 0
    total_rows = 0
    
    try:
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            # Adjust column index if needed (defaulting to col 0)
            for row in reader:
                if not row: continue
                text = " ".join(row) # Join all cols or pick specific one
                tokens = simple_tokenize(text)
                
                if not tokens: continue
                
                global_counts.update(tokens)
                if len(tokens) > 1:
                    # Store as "word1|word2" string for JSON compatibility
                    global_bigrams.update(f"{a}|{b}" for a, b in pairwise(tokens))
                
                current_doc.update(tokens)
                doc_size += 1
                total_rows += 1
                
                if doc_size >= 500: # Synthetic Doc Batch
                    topic_docs.append(dict(current_doc))
                    current_doc = collections.Counter()
                    doc_size = 0
                    if total_rows % 5000 == 0: print(f"Scanned {total_rows} rows...")

            # Flush
            if doc_size > 0: topic_docs.append(dict(current_doc))
            
    except Exception as e:
        print(f"Error: {e}")
        return

    # Output
    out_name = filename + "_sketch.json"
    output = {
        "total_rows": total_rows,
        "counts": dict(global_counts),
        "bigrams": dict(global_bigrams),
        "topic_docs": topic_docs
    }
    
    with open(out_name, 'w') as f:
        json.dump(output, f)
    
    print(f"Done! Saved to {out_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python harvest.py <filename.csv>")
    else:
        harvest(sys.argv[1])
    """, language="python")


# --- NEW: Process URLs and Manual Text into Virtual Files ---
all_inputs = list(uploaded_files) if uploaded_files else []

if url_input:
    urls_to_scrape = [u.strip() for u in url_input.split('\n') if u.strip()]
    if urls_to_scrape:
        with st.status(f"Scraping {len(urls_to_scrape)} URLs...", expanded=True) as status:
            for i, url in enumerate(urls_to_scrape):
                st.write(f"Fetching: {url}")
                scraped_text = fetch_url_content(url)
                if scraped_text:
                    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', url.split('//')[-1])[:50] + ".txt"
                    all_inputs.append(VirtualFile(safe_name, scraped_text))
            status.update(label="Scraping Complete", state="complete", expanded=False)

if manual_input:
    all_inputs.append(VirtualFile("manual_pasted_text.txt", manual_input))

if all_inputs:
    st.subheader("🚀 Scanning Phase")
    st.markdown("Configure how to read your files. Click 'Start Scan' to process them.")
    
    total_files = len(all_inputs)

    for idx, file in enumerate(all_inputs):
        file_bytes, fname, lower = file.getvalue(), file.name, file.name.lower()
        
        is_csv = lower.endswith(".csv")
        is_xlsx = lower.endswith((".xlsx", ".xlsm"))
        is_vtt = lower.endswith(".vtt")
        is_txt = lower.endswith(".txt")
        is_json = lower.endswith(".json")
        is_pdf = lower.endswith(".pdf")
        is_pptx = lower.endswith(".pptx")

        # --- input options w/ data preview-
        with st.expander(f"🧩 Scan Configuration: {fname}", expanded=True):
            if is_vtt: st.info("VTT transcript detected.")
            elif is_pdf: st.info("PDF detected.")
            elif is_pptx: st.info("PowerPoint detected.")
            elif is_txt: st.info("Plain Text detected.")
            elif is_csv:
                try: inferred_cols = detect_csv_num_cols(file_bytes, encoding_choice, delimiter=",")
                except Exception: inferred_cols = 1
                default_mode = "csv columns" if inferred_cols > 1 else "raw lines"
                read_mode = st.radio("read mode", ["raw lines", "csv columns"], index=0 if default_mode=="raw lines" else 1, key=f"csv_mode_{idx}")
                delim_choice = st.selectbox("delimiter", [",", "tab", ";", "|"], 0, key=f"csv_delim_{idx}")
                delimiter = {",": ",", "tab": "\t", ";": ";", "|": "|"}[delim_choice]
                has_header = st.checkbox("header row", value=True if inferred_cols > 1 else False, key=f"csv_header_{idx}")
                selected_cols, join_with = [], " "
                
                if read_mode == "csv columns":
                    st.caption("🔍 Data Preview")
                    df_prev = get_csv_preview(file_bytes, encoding_choice, delimiter, has_header)
                    st.dataframe(df_prev, use_container_width=True, height=150)
                    if not df_prev.empty:
                        col_names = list(df_prev.columns)
                        selected_cols = st.multiselect("Select Text Columns to Scan", col_names, [col_names[0]], key=f"csv_cols_{idx}")
                        join_with = st.text_input("join with", " ", key=f"csv_join_{idx}")
            elif is_xlsx:
                if openpyxl:
                    sheets = get_excel_sheetnames(file_bytes)
                    sheet_name = st.selectbox("sheet", sheets or ["(none)"], 0, key=f"xlsx_sheet_{idx}")
                    has_header = st.checkbox("header row", True, key=f"xlsx_header_{idx}")
                    if sheet_name:
                        st.caption("🔍 Data Preview")
                        df_prev = get_excel_preview(file_bytes, sheet_name, has_header)
                        st.dataframe(df_prev, use_container_width=True, height=150)
                        if not df_prev.empty:
                            col_names = list(df_prev.columns)
                            selected_cols = st.multiselect("Select Text Columns to Scan", col_names, [col_names[0]], key=f"xlsx_cols_{idx}")
                            join_with = st.text_input("join with", " ", key=f"xlsx_join_{idx}")

            elif is_json:
                st.info("JSON/JSONL File.")
                json_key = st.text_input("Key to Extract", "", key=f"json_key_{idx}")

        if st.button(f"⚡ Start Scan: {fname}", key=f"btn_scan_{idx}"):
            # Note: Batch size is already set by the sidebar logic logic, but repeating here is safe.
            st.session_state['sketch'].set_batch_size(doc_granularity)
            
            if clear_on_scan:
                reset_sketch() # Standard behavior: New Scan = New Analysis
                
            container = st.container()
            with container:
                per_file_bar, per_file_status = st.progress(0), st.empty()
            
            # --- PROGRESS RESET FIX ---
            per_file_bar.progress(0)
            per_file_status.empty()
            
            rows_iter, approx_rows = iter([]), 0
            
            # Setup Iterator (Streaming Read)
            if is_vtt:
                rows_iter = read_rows_vtt(file_bytes, "latin-1" if encoding_choice == "latin-1" else "auto")
                approx_rows = estimate_row_count_from_bytes(file_bytes)
            elif is_pdf:
                rows_iter = read_rows_pdf(file_bytes)
                approx_rows = 0
            elif is_pptx:
                rows_iter = read_rows_pptx(file_bytes)
                approx_rows = 0
            elif is_txt:
                rows_iter = read_rows_raw_lines(file_bytes, "latin-1" if encoding_choice == "latin-1" else "auto")
                approx_rows = estimate_row_count_from_bytes(file_bytes)
            elif is_json:
                key_sel = locals().get('json_key', None)
                rows_iter = read_rows_json(file_bytes, key_sel if key_sel and key_sel.strip() else None)
                approx_rows = estimate_row_count_from_bytes(file_bytes)
            elif is_csv:
                rmode = locals().get('read_mode', "raw lines")
                if rmode == "raw lines":
                    rows_iter = read_rows_raw_lines(file_bytes, "latin-1" if encoding_choice == "latin-1" else "auto")
                else:
                    rows_iter = iter_csv_selected_columns(file_bytes, "latin-1" if encoding_choice == "latin-1" else "auto", delimiter, has_header, selected_cols, join_with)
                approx_rows = estimate_row_count_from_bytes(file_bytes)
            elif is_xlsx and openpyxl:
                if sheet_name:
                    rows_iter = iter_excel_selected_columns(file_bytes, sheet_name, has_header, selected_cols, join_with)
                    approx_rows = excel_estimate_rows(file_bytes, sheet_name, has_header)
            else:
                rows_iter = read_rows_raw_lines(file_bytes, "latin-1" if encoding_choice == "latin-1" else "auto")
                approx_rows = estimate_row_count_from_bytes(file_bytes)
            
            start_wall = time.perf_counter()
            
            def make_progress_cb(total_hint: int):
                def _cb(done: int):
                    elapsed = time.perf_counter() - start_wall
                    if total_hint > 0:
                        per_file_bar.progress(min(99, int(done * 100 / total_hint)))
                        per_file_status.markdown(f"scanned rows: {done:,}/{total_hint:,} • {format_duration(elapsed)}")
                    else:
                        per_file_status.markdown(f"scanned rows: {done:,} • {format_duration(elapsed)}")
                return _cb
            
            # --- HYBRID VISUALIZATION LOGIC ---
            # We create a temp counter for THIS file only
            file_specific_stats = Counter()
            
            # Run scanner (updates st.session_state['sketch'] AND file_specific_stats)
            process_chunk_iter(
                rows_iter, remove_chat_artifacts, remove_html_tags, unescape_entities, remove_urls,
                keep_hyphens, keep_apostrophes, tuple(user_phrase_stopwords), tuple(user_single_stopwords),
                add_preps, drop_integers, min_word_len, compute_bigrams, st.session_state['sketch'],
                make_progress_cb(approx_rows), temp_file_stats=file_specific_stats
            )
            
            st.session_state['sketch'].finalize()
            per_file_bar.progress(100)
            per_file_status.success(f"Scan Complete! Rows added to Sketch.")
            
            # Warn if safety cap hit
            if st.session_state['sketch'].limit_reached:
                st.warning(f"⚠️ **Topic Modeling Cap Reached:** The analysis collected {st.session_state['sketch'].MAX_TOPIC_DOCS:,} document samples and then stopped adding more to prevent memory crash. Global word counts are still accurate, but topics may not reflect the very end of your data.")
            
            # RENDER IMMEDIATE FEEDBACK (Small File Mode)
            if file_specific_stats:
                st.markdown("##### 📄 Quick View: This File")
                color_func = None
                if enable_sentiment:
                    sentiments = get_sentiments(analyzer, tuple(file_specific_stats.keys()))
                    color_func = create_sentiment_color_func(sentiments, pos_color, neg_color, neu_color, pos_threshold, neg_threshold)
                
                fig, _ = build_wordcloud_figure_from_counts(file_specific_stats, max_words, width, height, bg_color, colormap, combined_font_path, random_state, color_func)
                col1, col2 = st.columns([3, 1])
                with col1: st.pyplot(fig, use_container_width=True)
                with col2: st.download_button(f"📥 download png", fig_to_png_bytes(fig), f"{fname}_wc.png", "image/png")
                plt.close(fig); del file_specific_stats; gc.collect()
            
            if not clear_on_scan:
                st.rerun()

# ----------------------------
# ANALYSIS PHASE (Reads from Sketch)
# ---------------------------
scanner = st.session_state['sketch']
combined_counts = scanner.global_counts
combined_bigrams = scanner.global_bigrams

if combined_counts:
    st.divider()
    st.header("📊 Analysis Phase")
    render_interpretation_guide() 
    # NEW: SKETCH EXPORT
    st.download_button(
        label="💾 Download Sketch (.json) for later",
        data=scanner.to_json(),
        file_name="data_sketch.json",
        mime="application/json",
        help="Save this sketch to skip scanning next time."
    )
    
    st.info(f"Analyzing Sketch of {scanner.total_rows_processed:,} total rows.")
    
    term_sentiments = {}
    if enable_sentiment:
        term_sentiments = get_sentiments(analyzer, tuple(combined_counts.keys()))
        if compute_bigrams:
            bigram_phrases = tuple(" ".join(bg) for bg in combined_bigrams.keys())
            term_sentiments.update(get_sentiments(analyzer, bigram_phrases))

    st.subheader("🔍 Bayesian Theme Discovery")
    # Extract model type from sidebar selection string
    selected_model_type = "LDA" if "LDA" in topic_model_type else "NMF"
    
    # --- IMPROVED EXPLANATION & TROUBLESHOOTING BLOCK ---
    with st.expander(f"🤔 How this works ({selected_model_type}) & Troubleshooting", expanded=False):
        n_docs = len(scanner.topic_docs)
        st.markdown(f"**Analysis Basis:** The model is learning from **{n_docs} synthetic documents** generated during the scan.")
        
        # Dynamic Warning for Low Resolution
        if n_docs < 10:
            st.warning(
                "⚠️ **Low Resolution Warning:** You have very few documents (data points) for the model to compare. "
                "Topic modeling works by finding contrasts *between* documents. If you only have 1 or 2, it often produces generic or repetitive topics.\n\n"
                "**Fix:** Go to the Sidebar ➔ Performance Options ➔ Set **'Rows per Document'** to **1** and re-scan."
            )

        if selected_model_type == "LDA":
            st.markdown("""
            **Latent Dirichlet Allocation (LDA)** creates a probabilistic model.
            *   **Logic:** It assumes every document is a "smoothie" of different ingredients (topics). It reads the documents to reverse-engineer the recipes.
            *   **Best For:** Long text, essays, assignments, and complex mixtures.
            """)
        else:
            st.markdown("""
            **Non-negative Matrix Factorization (NMF)** uses linear algebra.
            *   **Logic:** It forces text into distinct, sharp buckets. It assumes a document belongs to Category A *or* Category B, rarely both.
            *   **Best For:** Short chats, support tickets, and distinct feedback.
            """)
    # ----------------------------------------------------
    
    if len(scanner.topic_docs) > 0 and DictVectorizer:
        with st.spinner(f"Running {selected_model_type} Topic Modeling on Sketch..."):
            topics = perform_topic_modeling(scanner.topic_docs, n_topics=n_topics_val, model_type=selected_model_type)
        
        if topics:
            cols = st.columns(len(topics))
            for idx, topic in enumerate(topics):
                with cols[idx]:
                    st.markdown(f"**Topic {topic['id']}**")
                    for w in topic['words']:
                        st.markdown(f"`{w}`")
        else:
            st.warning("Not enough distinct data to detect topics.")
    
    st.divider()

    st.subheader("🖼️ Combined Word Cloud")
    try:
        c_color_func = None
        if enable_sentiment: c_color_func = create_sentiment_color_func(term_sentiments, pos_color, neg_color, neu_color, pos_threshold, neg_threshold)
        fig, _ = build_wordcloud_figure_from_counts(combined_counts, max_words, width, height, bg_color, colormap, combined_font_path, random_state, c_color_func)
        st.pyplot(fig, use_container_width=True)
        st.download_button("📥 download combined png", fig_to_png_bytes(fig), "combined_wc.png", "image/png")
        plt.close(fig); gc.collect()
    except MemoryError: st.error("memory error: reduce image size.")

    st.divider()
    
    text_stats = calculate_text_stats(combined_counts, scanner.total_rows_processed)

    show_graph = compute_bigrams and combined_bigrams and st.checkbox("🕸️ Show Network Graph & Advanced Analytics", value=True)
    
    # --- Bayesian Sentiment Inference
    if enable_sentiment and beta_dist:
        st.subheader("⚖️ Bayesian Sentiment Inference")
        bayes_result = perform_bayesian_sentiment_analysis(combined_counts, term_sentiments, pos_threshold, neg_threshold)
        if bayes_result:
            b_col1, b_col2 = st.columns([1, 2])
            with b_col1:
                st.metric("Positive Words Observed", f"{bayes_result['pos_count']:,}")
                st.metric("Negative Words Observed", f"{bayes_result['neg_count']:,}")
                st.info(f"Mean Expected Positive Rate: **{bayes_result['mean_prob']:.1%}**")
                st.success(f"95% Credible Interval:\n**{bayes_result['ci_low']:.1%} — {bayes_result['ci_high']:.1%}**")
            with b_col2:
                fig_bayes, ax_bayes = plt.subplots(figsize=(8, 4))
                ax_bayes.plot(bayes_result['x_axis'], bayes_result['pdf_y'], lw=2, color='blue', label='Posterior PDF')
                ax_bayes.fill_between(bayes_result['x_axis'], 0, bayes_result['pdf_y'], 
                                    where=(bayes_result['x_axis'] > bayes_result['ci_low']) & (bayes_result['x_axis'] < bayes_result['ci_high']),
                                    color='green', alpha=0.3, label='95% Credible Interval')
                ax_bayes.set_title("Bayesian Update of Sentiment Confidence", fontsize=10)
                ax_bayes.legend()
                ax_bayes.grid(True, alpha=0.2)
                st.pyplot(fig_bayes)
                plt.close(fig_bayes)
        st.divider()

    if show_graph:
        st.subheader("🔗 Network Graph & Analytics")
        # 1. graphing config/'physics'
        with st.expander("🛠️ Graph Settings & Physics", expanded=False):
            c1, c2, c3 = st.columns(3)
            # Default Min Link Frequency changed to 2 for better small-file performance
            min_edge_weight = c1.slider("Min Link Frequency", 2, 100, 2)
            max_nodes_graph = c1.slider("Max Nodes", 10, 200, 80)
            repulsion_val = c2.slider("Repulsion", 100, 3000, 1000)
            edge_len_val = c2.slider("Edge Length", 50, 500, 250)
            physics_enabled = c3.checkbox("Enable Physics", True)
            directed_graph = c3.checkbox("Directed Arrows", False)
            color_mode = c3.radio("Color By:", ["Community (Topic)", "Sentiment"], index=0)

        # 2. build graph
        G = nx.DiGraph() if directed_graph else nx.Graph()
        filtered_bigrams = {k: v for k, v in combined_bigrams.items() if v >= min_edge_weight}
        sorted_connections = sorted(filtered_bigrams.items(), key=lambda x: x[1], reverse=True)[:max_nodes_graph]
        
        if sorted_connections:
            for (source, target), weight in sorted_connections: G.add_edge(source, target, weight=weight)
            try: deg_centrality = nx.degree_centrality(G)
            except: deg_centrality = {n: 1 for n in G.nodes()}
            community_map = {}
            ai_cluster_info = "" 
            
            if color_mode == "Community (Topic)":
                G_undir = G.to_undirected() if directed_graph else G
                try:
                    communities = nx_comm.greedy_modularity_communities(G_undir)
                    cluster_descriptions = []
                    for group_id, community in enumerate(communities):
                        top_in_cluster = sorted(list(community), key=lambda x: combined_counts[x], reverse=True)[:5]
                        cluster_descriptions.append(f"- Cluster {group_id+1}: {', '.join(top_in_cluster)}")
                        for node in community: community_map[node] = group_id
                    ai_cluster_info = "\n".join(cluster_descriptions)
                except Exception as e: pass

            community_colors = ["#FF4B4B", "#4589ff", "#ffa421", "#3cdb82", "#8b46ff", "#ff4b9f", "#00c0f2"]
            nodes, edges = [], []
            for node_id in G.nodes():
                size = 15 + (deg_centrality.get(node_id, 0) * 80)
                if color_mode == "Sentiment":
                    node_color = neu_color
                    if enable_sentiment:
                        score = term_sentiments.get(node_id, 0)
                        if score >= pos_threshold: node_color = pos_color
                        elif score <= neg_threshold: node_color = neg_color
                else:
                    group_id = community_map.get(node_id, 0)
                    node_color = community_colors[group_id % len(community_colors)]

                nodes.append(Node(id=node_id, label=node_id, size=size, color=node_color, title=f"Term: {node_id}\nFreq: {combined_counts.get(node_id, 0)}", font={'color': 'white', 'size': 20, 'strokeWidth': 4, 'strokeColor': '#000000'}))

            for (source, target), weight in sorted_connections:
                width = 1 + math.log(weight) * 0.8
                edges.append(Edge(source=source, target=target, width=width, color="#e0e0e0"))
            
            config = Config(width=1000, height=700, directed=directed_graph, physics=physics_enabled, hierarchy=False, interaction={"navigationButtons": True, "zoomView": True}, physicsSettings={"solver": "forceAtlas2Based", "forceAtlas2Based": {"gravitationalConstant": -abs(repulsion_val), "springLength": edge_len_val, "springConstant": 0.05, "damping": 0.4}})
            st.info("💡 **Navigation Tip:** Use the buttons in the **bottom-right** of the graph to Zoom & Pan.")
            agraph(nodes=nodes, edges=edges, config=config)

            st.markdown("### 📊 Graph Analytics")
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Stats", "Top Nodes", "Text Stats", "🔥 Heatmap"])
            with tab1:
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Nodes", G.number_of_nodes())
                col_b2.metric("Edges", G.number_of_edges())
                try: col_b3.metric("Density", f"{nx.density(G):.4f}")
                except: pass
            with tab2:
                node_weights = {n: 0 for n in G.nodes()}
                for u, v, data in G.edges(data=True):
                    w = data.get('weight', 1)
                    node_weights[u] += w
                    node_weights[v] += w
                st.dataframe(pd.DataFrame(list(node_weights.items()), columns=["Node", "Weighted Degree"]).sort_values("Weighted Degree", ascending=False).head(50), use_container_width=True)
            with tab3:
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Total Tokens", f"{text_stats['Total Tokens']:,}")
                col_s2.metric("Unique Vocab", f"{text_stats['Unique Vocabulary']:,}")
                col_s3.metric("Lexical Diversity", f"{text_stats['Lexical Diversity']}")
                col_s4.metric("Avg Word Len", f"{text_stats['Avg Word Length']}")
            
            # --- NEW: HEATMAP VISUALIZATION ---
            with tab4:
                st.caption("Shows how often the Top 20 terms appear next to each other.")
                # 1. Get Top 20 terms
                top_20 = [w for w, c in combined_counts.most_common(20)]
                # 2. Build Matrix
                mat_size = len(top_20)
                heatmap_matrix = np.zeros((mat_size, mat_size))
                
                for i, w1 in enumerate(top_20):
                    for j, w2 in enumerate(top_20):
                        if i == j: 
                            heatmap_matrix[i][j] = 0 # No self loops in heatmap
                        else:
                            # Check (w1, w2) and (w2, w1)
                            val = combined_bigrams.get((w1, w2), 0) + combined_bigrams.get((w2, w1), 0)
                            heatmap_matrix[i][j] = val

                # 3. Plot
                if mat_size > 1:
                    fig_h, ax_h = plt.subplots(figsize=(10, 8))
                    im = ax_h.imshow(heatmap_matrix, cmap="Blues")
                    
                    # Labels
                    ax_h.set_xticks(np.arange(mat_size))
                    ax_h.set_yticks(np.arange(mat_size))
                    ax_h.set_xticklabels(top_20, rotation=45, ha="right")
                    ax_h.set_yticklabels(top_20)
                    
                    # Annotate
                    for i in range(mat_size):
                        for j in range(mat_size):
                            if heatmap_matrix[i, j] > 0:
                                ax_h.text(j, i, int(heatmap_matrix[i, j]), ha="center", va="center", color="black" if heatmap_matrix[i,j] < heatmap_matrix.max()/2 else "white", fontsize=8)
                    
                    ax_h.set_title("Top 20 Terms Co-occurrence")
                    plt.tight_layout()
                    st.pyplot(fig_h)
                    plt.close(fig_h)
                else:
                    st.info("Not enough data for heatmap.")

    else:
        st.subheader("📈 Text Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Total Tokens", f"{text_stats['Total Tokens']:,}")
        col_s2.metric("Unique Vocab", f"{text_stats['Unique Vocabulary']:,}")
        col_s3.metric("Lexical Diversity", f"{text_stats['Lexical Diversity']}")
        col_s4.metric("Avg Word Len", f"{text_stats['Avg Word Length']}")

else: 
    st.info("Uploaded data will be processed into the sketch above.")


# --- ai analytics section
if combined_counts and st.session_state['authenticated']:
    st.divider()
    st.subheader("🤖 AI Theme Detection")
    st.caption("Send the top 100 terms from the Sketch to the AI.")

    if st.button("✨ Analyze Themes with AI", type="primary"):
        with st.status("Analyzing top terms...", expanded=True) as status:
            g_context = locals().get("ai_cluster_info", "(Graph clustering not run)")
            response = generate_ai_insights(combined_counts, combined_bigrams if compute_bigrams else None, ai_config, g_context)
            st.session_state["ai_response"] = response
            status.update(label="Analysis Complete", state="complete", expanded=False)
            time.sleep(1.5) 
        st.rerun()

    if st.session_state.get("ai_response"):
        st.write(st.session_state["ai_response"])


if st.session_state['ai_response']:
    st.markdown("### 📋 AI Insights")
    st.markdown(st.session_state['ai_response'])
    st.divider()

# ---tables
if combined_counts:
    st.divider()
    st.subheader(f"📊 Frequency Tables (Top {top_n})")
    most_common = combined_counts.most_common(top_n)
    data = [[w, f] + ([term_sentiments.get(w,0), get_sentiment_category(term_sentiments.get(w,0), pos_threshold, neg_threshold)] if enable_sentiment else []) for w, f in most_common]
    cols = ["word", "count"] + (["sentiment", "category"] if enable_sentiment else [])
    st.dataframe(pd.DataFrame(data, columns=cols), use_container_width=True)
    
    if compute_bigrams and combined_bigrams:
        st.write("Bigrams (By Frequency)")
        top_bg = combined_bigrams.most_common(top_n)
        bg_data = [[" ".join(bg), f] + ([term_sentiments.get(" ".join(bg),0), get_sentiment_category(term_sentiments.get(" ".join(bg),0), pos_threshold, neg_threshold)] if enable_sentiment else []) for bg, f in top_bg]
        bg_cols = ["bigram", "count"] + (["sentiment", "category"] if enable_sentiment else [])
        st.dataframe(pd.DataFrame(bg_data, columns=bg_cols), use_container_width=True)

        # --- NEW: NPMI TABLE ---
        with st.expander("🔬 Phrase Significance (NPMI Score)", expanded=False):
            st.markdown("""
            **NPMI (Normalized Pointwise Mutual Information)** finds words that *belong* together, rather than just words that appear often.
            *   High Score (> 0.5): Strong association (e.g., "Artificial Intelligence").
            *   Low Score (< 0.1): Random association (e.g., "of the").
            """)
            df_npmi = calculate_npmi(combined_bigrams, combined_counts, scanner.total_rows_processed, min_freq=3)
            st.dataframe(df_npmi.head(top_n), use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #808080; font-size: 12px;'>"
    "Open Source software licensed under the MIT License."
    "</div>", 
    unsafe_allow_html=True
)
