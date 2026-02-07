"""Original function implementations from test.md for TDD comparison.

Each function is a standalone, self-contained implementation extracted from
well-known open-source projects. These serve as the "ground truth" that
the TDD-generated implementations will be compared against.
"""

import json
import re
from collections.abc import Sequence
from decimal import Decimal, getcontext, ROUND_HALF_EVEN, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, InvalidOperation
from difflib import SequenceMatcher


# ── 1. json_encode ────────────────────────────────────────────────
# Simplified from: cpython/Lib/json/encoder.py

def json_encode(o):
    """Encode a Python object into a JSON string.
    
    Supports: str, int, float, bool, None, dict, list, tuple.
    Raises TypeError for unsupported types or circular references.
    """
    if isinstance(o, str):
        # JSON string escaping
        s = o.replace('\\', '\\\\')
        s = s.replace('"', '\\"')
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return '"' + s + '"'
    if o is None:
        return 'null'
    if o is True:
        return 'true'
    if o is False:
        return 'false'
    if isinstance(o, int):
        return str(o)
    if isinstance(o, float):
        if o != o:  # NaN
            raise ValueError("Out of range float values are not JSON compliant")
        if o == float('inf') or o == float('-inf'):
            raise ValueError("Out of range float values are not JSON compliant")
        return repr(o)
    if isinstance(o, dict):
        items = []
        for k, v in o.items():
            if isinstance(k, str):
                key = json_encode(k)
            elif isinstance(k, (int, float)):
                key = json_encode(str(k))
            elif k is None:
                key = '"null"'
            elif isinstance(k, bool):
                key = json_encode(str(k).lower())
            else:
                raise TypeError(f"keys must be str, int, float, bool or None, not {type(k).__name__}")
            items.append(key + ':' + json_encode(v))
        return '{' + ','.join(items) + '}'
    if isinstance(o, (list, tuple)):
        return '[' + ','.join(json_encode(i) for i in o) + ']'
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# ── 2. slugify ────────────────────────────────────────────────────
# Simplified from: python-slugify

def slugify(text: str, max_length: int = 0, word_boundary: bool = False,
            separator: str = '-', stopwords: tuple = (), 
            replacements: tuple = (), lowercase: bool = True,
            allow_unicode: bool = False) -> str:
    """Convert a string to a URL-safe slug.
    
    Transliterates unicode to ASCII by default, strips special chars,
    replaces spaces with separator.
    """
    text = str(text)
    for a, b in replacements:
        text = text.replace(a, b)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Remove stopwords
    if stopwords:
        pattern = r'\b(' + '|'.join(map(re.escape, stopwords)) + r')\b'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
    if lowercase:
        text = text.lower()
    if not allow_unicode:
        # Simple transliteration for common chars
        _map = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'Ç': 'C', 'Ğ': 'G', 'İ': 'I', 'Ö': 'O', 'Ş': 'S', 'Ü': 'U',
            'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
            'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
            'ß': 'ss', 'ñ': 'n', 'æ': 'ae', 'œ': 'oe',
        }
        text = ''.join(_map.get(c, c) for c in text)
        # Remove non-word chars (keep alphanumeric, hyphen, underscore)
        text = re.sub(r'[^\w\s-]', '', text)
    else:
        text = re.sub(r'[^\w\s-]', '', text)
    # Collapse runs of separator/whitespace
    text = re.sub(r'[-\s]+', separator, text).strip(separator)
    if max_length > 0:
        text = text[:max_length].rstrip(separator)
        if word_boundary:
            i = text.rfind(separator)
            if 0 < i < max_length:
                text = text[:i]
    return text


# ── 3. levenshtein ────────────────────────────────────────────────
# Classic DP implementation

def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.
    
    Returns the minimum number of single-character edits (insertions,
    deletions, substitutions) required to change s1 into s2.
    Raises TypeError for non-string inputs.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Both arguments must be strings")
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# ── 4. ordinal ────────────────────────────────────────────────────
# From: humanize/number.py

def ordinal(value):
    """Convert a number to its ordinal string representation.
    
    E.g. 1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th'.
    Non-numeric inputs are returned as-is.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if 10 <= abs(value) % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(abs(value) % 10, 'th')
    return str(value) + suffix


# ── 5. smart_str ──────────────────────────────────────────────────
# From: django/utils/encoding.py

def smart_str(s, encoding="utf-8", strings_only=False, errors="strict"):
    """Convert input to str safely with encoding support.
    
    If strings_only=True, non-string types are returned as-is.
    Bytes are decoded with the given encoding.
    Other types are converted via str().
    """
    _PROTECTED_TYPES = (int, float, bool, type(None), list, dict, tuple, set)
    if strings_only and isinstance(s, _PROTECTED_TYPES):
        return s
    if isinstance(s, bytes):
        return str(s, encoding, errors)
    elif isinstance(s, str):
        return s
    else:
        try:
            return str(s)
        except UnicodeEncodeError:
            if isinstance(s, Exception):
                return " ".join(smart_str(arg, encoding, strings_only, errors) for arg in s.args)
            return str(s).encode(encoding, errors).decode(encoding)


# ── 6. get_close_matches ─────────────────────────────────────────
# From: cpython/Lib/difflib.py

def get_close_matches(word, possibilities, n=3, cutoff=0.6):
    """Find close matches of word in a list of possibilities.
    
    Returns at most n items from possibilities that are sufficiently
    similar to word. cutoff (0-1) controls the similarity threshold.
    """
    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for x in possibilities:
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result.append((s.ratio(), x))
    result = sorted(result, key=lambda x: x[0], reverse=True)
    return [x for score, x in result[:n]]


# ── 7. decimal_quantize ──────────────────────────────────────────
# Wrapper around stdlib Decimal.quantize

def decimal_quantize(value, exp, rounding=None):
    """Quantize a Decimal value to the given exponent.
    
    Args:
        value: Decimal value (or string/number convertible to Decimal)
        exp: Exponent to quantize to (e.g. '0.01' for 2 decimal places)
        rounding: Rounding mode (e.g. ROUND_HALF_UP, ROUND_DOWN)
    
    Returns:
        Quantized Decimal value
    """
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not isinstance(exp, Decimal):
        exp = Decimal(str(exp))
    if rounding:
        return value.quantize(exp, rounding=rounding)
    return value.quantize(exp)


# ── Registry ──────────────────────────────────────────────────────

FUNCTIONS = {
    "json_encode": {
        "func": json_encode,
        "description": (
            "Encode a Python object into a JSON string. "
            "Supports str, int, float, bool, None, dict, list, tuple. "
            "Dict keys that are int/float/bool/None are coerced to strings. "
            "Raises TypeError for unsupported types (set, object, lambda). "
            "Raises ValueError for float('inf') and float('nan'). "
            "Tuples are serialized as JSON arrays. "
            "Strings are properly escaped (newlines, tabs, quotes, backslashes)."
        ),
    },
    "slugify": {
        "func": slugify,
        "description": (
            "Convert a text string to a URL-safe slug. "
            "Parameters: text (str), max_length (int, default 0 = unlimited), "
            "word_boundary (bool), separator (str, default '-'), "
            "stopwords (tuple of words to remove), replacements (tuple of (old,new) pairs), "
            "lowercase (bool, default True), allow_unicode (bool, default False). "
            "By default: lowercases, transliterates common unicode chars to ASCII, "
            "strips non-alphanumeric chars, collapses whitespace/hyphens to separator."
        ),
    },
    "levenshtein": {
        "func": levenshtein,
        "description": (
            "Compute the Levenshtein (edit) distance between two strings. "
            "Returns the minimum number of single-character edits "
            "(insertions, deletions, substitutions) to transform s1 into s2. "
            "Both arguments must be strings, raises TypeError otherwise. "
            "Returns 0 for identical strings, len(s1) if s2 is empty."
        ),
    },
    "ordinal": {
        "func": ordinal,
        "description": (
            "Convert a number to its English ordinal string. "
            "1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', "
            "11 -> '11th', 12 -> '12th', 13 -> '13th' (teen special case), "
            "21 -> '21st', 101 -> '101st'. "
            "Accepts int, float (truncated), or numeric strings. "
            "Non-numeric inputs (None, 'abc', [], etc.) are returned unchanged."
        ),
    },
    "smart_str": {
        "func": smart_str,
        "description": (
            "Safely convert any input to str with encoding support. "
            "Parameters: s (input), encoding (default 'utf-8'), "
            "strings_only (bool, default False), errors (default 'strict'). "
            "Bytes are decoded with the given encoding. "
            "If strings_only=True, protected types (int, float, bool, None, "
            "list, dict, tuple, set) are returned as-is without conversion. "
            "Strings pass through unchanged. Other types use str()."
        ),
    },
    "get_close_matches": {
        "func": get_close_matches,
        "description": (
            "Find the closest matching strings from a list of possibilities. "
            "Parameters: word (str to match), possibilities (list of candidate strings), "
            "n (max results, default 3), cutoff (similarity threshold 0.0-1.0, default 0.6). "
            "Returns up to n strings sorted by similarity (best first). "
            "Uses SequenceMatcher ratio. Raises ValueError if n<=0 or cutoff out of [0,1]."
        ),
    },
    "decimal_quantize": {
        "func": decimal_quantize,
        "description": (
            "Quantize (round) a Decimal value to a given precision. "
            "Parameters: value (Decimal or str/number), exp (Decimal or str like '0.01'), "
            "rounding (optional rounding mode like ROUND_HALF_UP, ROUND_DOWN). "
            "Converts non-Decimal inputs to Decimal via str(). "
            "Returns the quantized Decimal. Follows standard Decimal quantize rules."
        ),
    },
}
