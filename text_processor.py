import re
import html
import string
import csv
import io
import json
import pandas as pd
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Optional, Callable
from itertools import pairwise
from wordcloud import STOPWORDS
import openpyxl

# Patterns
HTML_TAG_RE = re.compile(r"<[^>]+>")
CHAT_ARTIFACT_RE = re.compile(
    r":\w+:"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|yesterday) at \d{1,2}:\d{2}\b"
    r"|\b\d+\s+repl(?:y|ies)\b"
    r"|\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
    r"|\[[^\]]+\]",
    flags=re.IGNORECASE
)

def default_prepositions() -> set:
    return {'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'but', 'by', 'concerning', 'despite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'regarding', 'since', 'through', 'throughout', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without'}

def build_punct_translation(keep_hyphens: bool, keep_apostrophes: bool) -> dict:
    punct = string.punctuation
    if keep_hyphens: punct = punct.replace("-", "")
    if keep_apostrophes: punct = punct.replace("'", "")
    return str.maketrans("", "", punct)

def is_url_token(tok: str) -> bool:
    t = tok.strip("()[]{}<>,.;:'\"!?").lower()
    if not t: return False
    return ("://" in t) or t.startswith("www.")

def clean_and_tokenize(
    text: str,
    remove_chat: bool, remove_html: bool, unescape: bool, remove_urls: bool,
    trans_map: dict,
    stopwords: set,
    phrase_pattern: Optional[re.Pattern],
    min_len: int, drop_int: bool
) -> List[str]:
    """Core stateless tokenizer used by both App and Harvester."""
    if not text: return []
    if remove_chat: text = CHAT_ARTIFACT_RE.sub(" ", text)
    if remove_html: text = HTML_TAG_RE.sub(" ", text)
    if unescape:
        try: text = html.unescape(text)
        except: pass
    text = text.lower()
    if phrase_pattern: text = phrase_pattern.sub(" ", text)
    
    tokens = []
    for t in text.split():
        if remove_urls and is_url_token(t): continue
        t = t.translate(trans_map)
        if not t or len(t) < min_len or (drop_int and t.isdigit()) or t in stopwords: continue
        tokens.append(t)
    return tokens

def get_csv_columns_fast(file_path_or_buffer, encoding="utf-8", delimiter=",") -> List[str]:
    """Quickly grab CSV headers."""
    try:
        if isinstance(file_path_or_buffer, str):
            with open(file_path_or_buffer, 'r', encoding=encoding, errors='replace') as f:
                reader = csv.reader(f, delimiter=delimiter)
                return next(reader, [])
        else:
            # BytesIO buffer
            wrapper = io.TextIOWrapper(file_path_or_buffer, encoding=encoding, errors='replace', newline="")
            reader = csv.reader(wrapper, delimiter=delimiter)
            row = next(reader, [])
            wrapper.detach() # Don't close the BytesIO
            return row
    except:
        return []
