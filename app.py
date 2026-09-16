"""
TrustScan | Fake & Misleading E-Commerce Review Detector
Interactive Streamlit Prototype using Heuristic NLP Analysis
"""

import math
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st
from textblob import TextBlob

# -----------------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrustScan | Fake Review NLP Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, production-grade UI
st.markdown(
    """
    <style>
    /* Metric Cards */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .trust-hero-card {
        border-radius: 14px;
        padding: 24px;
        margin: 10px 0 20px 0;
        text-align: center;
        border: 2px solid;
    }
    .trust-hero-card.authentic {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(16, 185, 129, 0.05));
        border-color: #22c55e;
    }
    .trust-hero-card.suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(217, 119, 6, 0.05));
        border-color: #f59e0b;
    }
    .trust-hero-card.fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(220, 38, 38, 0.05));
        border-color: #ef4444;
    }
    .trust-score-num {
        font-size: 3.8rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }
    .trust-verdict-title {
        font-size: 1.35rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }
    .tag-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .tag-pill.pass { background-color: #dcfce7; color: #166534; }
    .tag-pill.warn { background-color: #fef3c7; color: #92400e; }
    .tag-pill.fail { background-color: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# CURATED PRESET SAMPLES
# -----------------------------------------------------------------------------
SAMPLE_REVIEWS: Dict[str, Dict[str, Any]] = {
    "🤖 Bot Copy-Paste Spam (Repetitive Keywords & Low TTR)": {
        "rating": 5,
        "text": (
            "Best bluetooth speaker ever! Buy this bluetooth speaker now. "
            "This bluetooth speaker is great. Very good speaker. Best speaker bluetooth. "
            "Buy speaker now. Great bluetooth speaker quality. Best bluetooth speaker ever! "
            "Buy this bluetooth speaker now. Best bluetooth speaker."
        ),
        "desc": "Keyword stuffing and copy-paste phrase repetition designed to game search ranking.",
    },
    "⚡ Inverted Rating Glitch (5-Star Rating with Scathing Negative Text)": {
        "rating": 5,
        "text": (
            "Absolute garbage. The zipper broke on day one, customer support never responded, "
            "and the fabric tore after a single wash. Completely useless and a total waste of money. "
            "Do not buy this defective trash!"
        ),
        "desc": "Blatant divergence: 5-star rating assigned to an intensely negative review (often bot farm error or sarcasm).",
    },
    "📢 Paid Hype Fluff (Superlatives, Excessive Caps & Exclamations, No Specs)": {
        "rating": 5,
        "text": (
            "OMG BEST PRODUCT EVER IN MY ENTIRE LIFE!!!!!! THIS IS A MIRACLE PRODUCT! "
            "I AM SOOOOO IN LOVE WITH THIS AMAZING THING! MUST BUY NOW EVERYONE! "
            "100% PERFECT AND FLAWLESS! CHANGED MY LIFE COMPLETELY! WOW WOW WOW!!!!!!"
        ),
        "desc": "Overheated marketing fluff filled with ALL-CAPS, punctuation runs, and zero concrete nouns or specs.",
    },
    "✅ Genuine Authentic Review (Balanced, Concrete Details & Specs)": {
        "rating": 4,
        "text": (
            "I have been using this wireless mouse for about 3 weeks with my MacBook Pro. "
            "The Bluetooth pairing took under 15 seconds and battery life has been solid—still at 78% "
            "after daily 8-hour office use. The click mechanism is quiet and the ergonomic shape "
            "fits medium hands comfortably. My only minor complaint is that the USB-C charging port "
            "is placed awkwardly on the front lip, making it awkward to use while charging."
        ),
        "desc": "Organic customer review with nuanced pros/cons, concrete nouns, and realistic usage specifications.",
    },
    "📉 Genuine Critical Review (1-Star, Detailed Factual Defect)": {
        "rating": 1,
        "text": (
            "Ordered the 32GB model in matte black. The device arrived on Friday, but the power button "
            "was sunken and wouldn't click. When plugged into the standard 20W charger, the unit "
            "heated up to an uncomfortable degree within 10 minutes and never turned on. "
            "Initiated a return on Saturday morning. Very disappointing build quality for $180."
        ),
        "desc": "Legitimate negative review with specific product specs, dates, and objective defect descriptions.",
    },
    "✏️ Blank / Custom Input": {
        "rating": 5,
        "text": "",
        "desc": "Paste your own e-commerce review text and select a star rating.",
    },
}

# -----------------------------------------------------------------------------
# NLP HEURISTICS ENGINE
# -----------------------------------------------------------------------------

CRITICAL_NEGATIVE_TERMS = {
    "garbage", "trash", "defective", "scam", "broken", "broke", "ripoff",
    "terrible", "awful", "horrible", "unusable", "fraud", "useless", "disaster",
    "worst", "regret", "counterfeit", "fake", "junk"
}

CRITICAL_POSITIVE_TERMS = {
    "excellent", "fantastic", "amazing", "perfect", "flawless", "superb",
    "outstanding", "brilliant", "wonderful", "stellar", "love", "loved", "awesome", "great"
}

SUPERLATIVE_PATTERNS = [
    r"\bbest\s+(ever|thing|product|purchase|decision)\b",
    r"\bmiracle\b",
    r"\blife\s+changing\b",
    r"\bunbelievable\b",
    r"\bholy\s+grail\b",
    r"\bmust\s+buy\b",
    r"\b100%\s+recommend\b",
    r"\b10/10\b",
    r"\bblown\s+away\b",
    r"\bgame\s+changer\b",
    r"\bperfect(ion|ly)?\b",
    r"\bflawless(ly)?\b",
    r"\brevolutionary\b",
    r"\bgreatest\b",
    r"\bobsessed\b",
    r"\btotal\s+scam\b",
    r"\bworst\s+ever\b",
]

SPEC_PATTERNS = [
    r"\b\d+(\.\d+)?\s*(hours?|hrs?|days?|weeks?|months?|years?|oz|lbs?|kg|g|inches?|in|cm|mm|watts?|w|hz|mah|gb|mb|tb|k|v|volts?)\b",
    r"\$\d+(\.\d+)?",
    r"\b(battery|screen|display|charger|cable|port|zipper|fabric|plastic|metal|aluminum|leather|fit|size|color|weight|latency|audio|bass|volume|shipping|delivery|packaging|instructions|warranty)\b",
]

COMMON_ACRONYMS = {
    "USB", "LED", "TV", "PC", "RAM", "SSD", "HDD", "GPU", "CPU", "RGB",
    "HDMI", "OK", "AC", "DC", "ID", "UK", "US", "USA", "EU", "AM", "PM", "APP"
}


def clean_tokenize(text: str) -> List[str]:
    """Tokenize words cleanly using regular expressions."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def analyze_sentiment_divergence(
    text: str, rating: int
) -> Dict[str, Any]:
    """
    Evaluates alignment between declared star rating (1-5) and text sentiment polarity.
    - Uses TextBlob sentiment augmented with e-commerce domain lexicons.
    - Normalizes rating 1-5 to [-1.0, +1.0].
    - Detects severe rating vs. text contradictions.
    """
    if not text.strip():
        return {
            "expected_polarity": 0.0,
            "actual_polarity": 0.0,
            "divergence": 0.0,
            "penalty": 0.0,
            "flag": "Pass",
            "summary": "No text to analyze",
        }

    blob = TextBlob(text)
    base_polarity = blob.sentiment.polarity

    # Domain keyword booster (catches e-commerce specific terms not in standard TextBlob)
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
    neg_hits = len(words.intersection(CRITICAL_NEGATIVE_TERMS))
    pos_hits = len(words.intersection(CRITICAL_POSITIVE_TERMS))

    adjusted_polarity = base_polarity
    if neg_hits > 0:
        adjusted_polarity = min(adjusted_polarity, base_polarity - neg_hits * 0.15)
    if pos_hits > 0:
        adjusted_polarity = max(adjusted_polarity, base_polarity + pos_hits * 0.12)

    actual_polarity = round(max(-1.0, min(1.0, adjusted_polarity)), 3)

    # Expected polarity: 1 -> -1.0, 2 -> -0.5, 3 -> 0.0, 4 -> +0.5, 5 -> +1.0
    expected_polarity = round((rating - 3.0) / 2.0, 3)

    # Normalized divergence between [-1, 1] is max 2.0
    raw_diff = abs(expected_polarity - actual_polarity)
    normalized_divergence = round(raw_diff / 2.0, 3)

    # Severe mismatch conditions (e.g., 5-star with negative text or 1-star with glowing praise)
    is_severe_mismatch = (
        (rating == 5 and actual_polarity < -0.20)
        or (rating >= 4 and actual_polarity < -0.45)
        or (rating == 1 and actual_polarity > 0.25)
        or (rating <= 2 and actual_polarity > 0.45)
    )

    # Natural leeway: reviews within 0.38 divergence are natural human variance
    tolerance = 0.38
    if normalized_divergence <= tolerance:
        penalty = 0.0
    else:
        penalty = min(100.0, ((normalized_divergence - tolerance) / (1.0 - tolerance)) ** 1.2 * 100.0)

    if is_severe_mismatch:
        penalty = max(penalty, 85.0)

    penalty = round(penalty, 1)

    if penalty >= 60:
        flag = "Flagged"
        summary = f"Severe contradiction! {rating}★ rating contradicts {actual_polarity:+.2f} text polarity."
    elif penalty >= 30:
        flag = "Warning"
        summary = f"Moderate divergence between rating ({rating}★) and text sentiment ({actual_polarity:+.2f})."
    else:
        flag = "Pass"
        summary = f"Consistent: {rating}★ aligns well with {actual_polarity:+.2f} polarity."

    return {
        "expected_polarity": expected_polarity,
        "actual_polarity": actual_polarity,
        "divergence": normalized_divergence,
        "penalty": penalty,
        "flag": flag,
        "summary": summary,
    }


def analyze_lexical_diversity(
    text: str, ttr_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Evaluates Type-Token Ratio (TTR) and repetitive n-grams.
    - Low TTR indicates copy-paste spam or bot keyword stuffing.
    - Calculates both standard TTR and Root TTR (length-stabilized).
    """
    tokens = clean_tokenize(text)
    total_tokens = len(tokens)

    if total_tokens < 6:
        return {
            "total_tokens": total_tokens,
            "unique_tokens": len(set(tokens)),
            "ttr": 1.0,
            "root_ttr": round(math.sqrt(total_tokens), 2) if total_tokens > 0 else 0.0,
            "penalty": 0.0,
            "repeated_trigrams": 0,
            "flag": "Pass",
            "summary": "Review too short for meaningful lexical diversity penalty.",
        }

    unique_tokens = len(set(tokens))
    ttr = round(unique_tokens / total_tokens, 3)
    root_ttr = round(unique_tokens / math.sqrt(total_tokens), 2)

    # Detect repeated trigrams (3 consecutive words repeating)
    trigrams = [tuple(tokens[i : i + 3]) for i in range(total_tokens - 2)]
    trigram_counts: Dict[Tuple[str, ...], int] = {}
    for tg in trigrams:
        trigram_counts[tg] = trigram_counts.get(tg, 0) + 1
    repeated_trigrams = sum(1 for count in trigram_counts.values() if count > 1)

    # Penalty scoring
    if ttr >= ttr_threshold:
        base_penalty = 0.0
    else:
        floor = 0.25
        deficit = (ttr_threshold - max(ttr, floor)) / (ttr_threshold - floor)
        base_penalty = deficit * 85.0

    trigram_penalty = min(25.0, repeated_trigrams * 10.0)
    penalty = round(min(100.0, base_penalty + trigram_penalty), 1)

    if penalty >= 60:
        flag = "Flagged"
        summary = f"Repetitive bot vocabulary: TTR is only {ttr:.2f} with {repeated_trigrams} repeated trigrams."
    elif penalty >= 30:
        flag = "Warning"
        summary = f"Below-average diversity (TTR: {ttr:.2f}). Noticeable keyword repetition."
    else:
        flag = "Pass"
        summary = f"Natural lexical diversity (TTR: {ttr:.2f}, Root TTR: {root_ttr:.1f})."

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "repeated_trigrams": repeated_trigrams,
        "penalty": penalty,
        "flag": flag,
        "summary": summary,
    }


def analyze_subjectivity(text: str) -> Dict[str, Any]:
    """
    Calculates TextBlob subjectivity and balances emotional superlatives
    against concrete product nouns and technical specifications.
    """
    if not text.strip():
        return {
            "subjectivity": 0.0,
            "superlatives_count": 0,
            "specs_count": 0,
            "penalty": 0.0,
            "flag": "Pass",
            "summary": "No text provided.",
            "found_superlatives": [],
        }

    blob = TextBlob(text)
    subjectivity = round(blob.sentiment.subjectivity, 3)

    lower_text = text.lower()

    # Find superlatives
    found_superlatives = []
    for pattern in SUPERLATIVE_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            found_superlatives.extend(matches)
    superlatives_count = len(found_superlatives)

    # Find concrete specifications / product terms
    found_specs = []
    for pattern in SPEC_PATTERNS:
        matches = re.findall(pattern, lower_text)
        if matches:
            found_specs.extend(matches)
    specs_count = len(found_specs)

    # Score calculation:
    # High subjectivity (>0.55) + superlatives - concrete specs discount
    base_sub_penalty = max(0.0, (subjectivity - 0.55) / 0.45) * 50.0
    superlative_penalty = min(40.0, superlatives_count * 15.0)
    spec_discount = min(30.0, specs_count * 8.0)

    raw_penalty = base_sub_penalty + superlative_penalty - spec_discount
    penalty = round(max(0.0, min(100.0, raw_penalty)), 1)

    if penalty >= 60:
        flag = "Flagged"
        summary = f"Overheated hype! High subjectivity ({subjectivity:.2f}) with {superlatives_count} superlatives and zero concrete product specs."
    elif penalty >= 30:
        flag = "Warning"
        summary = f"Elevated subjectivity ({subjectivity:.2f}) with {superlatives_count} superlative phrases."
    else:
        flag = "Pass"
        summary = f"Grounded content: Balanced subjectivity ({subjectivity:.2f}) with {specs_count} concrete specifications."

    return {
        "subjectivity": subjectivity,
        "superlatives_count": superlatives_count,
        "specs_count": specs_count,
        "found_superlatives": list(set(found_superlatives)),
        "penalty": penalty,
        "flag": flag,
        "summary": summary,
    }


def analyze_typographical_anomalies(
    text: str, max_caps_ratio: float = 0.15, max_punct_density: float = 0.04
) -> Dict[str, Any]:
    """
    Detects typographical anomalies:
    - ALL-CAPS words ratio (ignoring single letters & common technical acronyms)
    - Excessive punctuation (runs of !!! or ???, high exclamation density)
    - Elongated words (e.g. 'sooooo', 'greeeeat')
    """
    if not text.strip():
        return {
            "caps_ratio": 0.0,
            "punct_density": 0.0,
            "excessive_punct_runs": 0,
            "elongated_words": 0,
            "penalty": 0.0,
            "flag": "Pass",
            "summary": "No text provided.",
            "caps_words": [],
        }

    raw_words = re.findall(r"\b[a-zA-Z]{2,}\b", text)
    total_words = len(raw_words)

    if total_words == 0:
        caps_ratio = 0.0
        caps_words = []
    else:
        caps_words = [
            w for w in raw_words if w.isupper() and w not in COMMON_ACRONYMS
        ]
        caps_ratio = round(len(caps_words) / total_words, 3)

    # Punctuation analysis
    exclamations = text.count("!")
    question_marks = text.count("?")
    total_chars = max(len(text), 1)
    punct_density = round((exclamations + question_marks) / total_chars, 3)

    # Runs of 2 or more punctuation marks (e.g., !!, ???, !?)
    punct_runs = re.findall(r"[!?]{2,}", text)
    excessive_punct_runs = len(punct_runs)

    # Character elongations (e.g. 'loooove', 'soooo')
    elongated = re.findall(r"([a-zA-Z])\1{2,}", text)
    elongated_words_count = len(elongated)

    # Penalty calculation
    caps_penalty = 0.0
    if caps_ratio > max_caps_ratio:
        caps_penalty = min(50.0, ((caps_ratio - max_caps_ratio) / (0.60 - max_caps_ratio)) * 50.0)

    punct_penalty = 0.0
    if punct_density > max_punct_density:
        punct_penalty = min(35.0, ((punct_density - max_punct_density) / 0.10) * 35.0)

    run_penalty = min(25.0, excessive_punct_runs * 8.0 + elongated_words_count * 5.0)

    penalty = round(min(100.0, caps_penalty + punct_penalty + run_penalty), 1)

    if penalty >= 60:
        flag = "Flagged"
        summary = f"High typographical anomalies: {caps_ratio*100:.1f}% ALL-CAPS words, {excessive_punct_runs} punctuation runs."
    elif penalty >= 25:
        flag = "Warning"
        summary = f"Elevated capitalization or punctuation usage (Caps: {caps_ratio*100:.1f}%, Runs: {excessive_punct_runs})."
    else:
        flag = "Pass"
        summary = f"Normal typography: {caps_ratio*100:.1f}% uppercase, standard punctuation."

    return {
        "caps_ratio": caps_ratio,
        "caps_words_count": len(caps_words),
        "caps_words": caps_words[:8],
        "punct_density": punct_density,
        "excessive_punct_runs": excessive_punct_runs,
        "punct_runs": punct_runs[:5],
        "elongated_words": elongated_words_count,
        "penalty": penalty,
        "flag": flag,
        "summary": summary,
    }


def compute_composite_trust_score(
    sentiment_res: Dict[str, Any],
    lexical_res: Dict[str, Any],
    subjectivity_res: Dict[str, Any],
    typo_res: Dict[str, Any],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Computes a blended composite Trust Score (0-100%) and classification verdict.
    Blends weighted average penalties with peak penalty safeguard to prevent
    critical red flags from being diluted by neutral heuristics.
    """
    # Normalize weights to sum to 1.0
    total_w = sum(weights.values())
    norm_w = {k: v / total_w for k, v in weights.items()} if total_w > 0 else {k: 0.25 for k in weights}

    # Weighted penalty
    weighted_penalty = (
        norm_w["sentiment"] * sentiment_res["penalty"]
        + norm_w["lexical"] * lexical_res["penalty"]
        + norm_w["subjectivity"] * subjectivity_res["penalty"]
        + norm_w["typography"] * typo_res["penalty"]
    )

    # Anomaly peak safeguard (an extreme red flag in one dimension should not be masked)
    peak_penalty = max(
        sentiment_res["penalty"],
        lexical_res["penalty"],
        subjectivity_res["penalty"],
        typo_res["penalty"],
    )

    # 55% weighted average + 45% peak penalty safeguard
    blended_penalty = 0.55 * weighted_penalty + 0.45 * peak_penalty
    trust_score = round(max(0.0, min(100.0, 100.0 - blended_penalty)), 1)

    if trust_score >= 80.0:
        verdict = "Authentic & Organic"
        verdict_class = "authentic"
        badge_icon = "🟢"
        explanation = (
            "This review exhibits healthy lexical diversity, realistic phrasing, "
            "and natural alignment between the star rating and sentiment."
        )
    elif trust_score >= 50.0:
        verdict = "Moderately Suspicious"
        verdict_class = "suspicious"
        badge_icon = "🟡"
        explanation = (
            "This review shows unnatural patterns, such as repetitive vocabulary, "
            "elevated subjectivity, or mild sentiment-rating divergence."
        )
    else:
        verdict = "High Risk / Likely Fake or Automated"
        verdict_class = "fake"
        badge_icon = "🔴"
        explanation = (
            "Multiple critical heuristics were flagged. The review strongly correlates "
            "with bot-generated spam, paid hype fluff, or glitched review farming."
        )

    return {
        "trust_score": trust_score,
        "weighted_penalty": round(blended_penalty, 1),
        "raw_weighted_penalty": round(weighted_penalty, 1),
        "peak_penalty": round(peak_penalty, 1),
        "verdict": verdict,
        "verdict_class": verdict_class,
        "badge_icon": badge_icon,
        "explanation": explanation,
        "normalized_weights": norm_w,
    }


# -----------------------------------------------------------------------------
# STREAMLIT UI LAYOUT
# -----------------------------------------------------------------------------

def main():
    # Header Banner
    st.title("🛡️ TrustScan: Review Credibility Analyzer")
    st.caption(
        "Real-time heuristic NLP engine detecting automated, manipulated, or misleading e-commerce reviews."
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: CONFIGURATION & WEIGHT TUNING
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Heuristics & Weights")
        st.markdown(
            "Adjust the strictness and contribution of each NLP heuristic to suit different e-commerce verticals."
        )

        preset_mode = st.selectbox(
            "Preset Profile",
            options=[
                "Balanced (Default)",
                "Strict Bot Hunter (Focus on Repetition & Caps)",
                "Hype & Fluff Filter (Focus on Subjectivity & Claims)",
                "Sentiment Auditor (Focus on Star-Text Consistency)",
                "Custom",
            ],
            index=0,
            help="Select a tuned profile or choose Custom to manually adjust weights.",
        )

        # Default weights based on preset
        if preset_mode == "Balanced (Default)":
            w_sent, w_lex, w_subj, w_typo = 0.35, 0.25, 0.20, 0.20
        elif preset_mode == "Strict Bot Hunter (Focus on Repetition & Caps)":
            w_sent, w_lex, w_subj, w_typo = 0.20, 0.40, 0.15, 0.25
        elif preset_mode == "Hype & Fluff Filter (Focus on Subjectivity & Claims)":
            w_sent, w_lex, w_subj, w_typo = 0.20, 0.20, 0.45, 0.15
        elif preset_mode == "Sentiment Auditor (Focus on Star-Text Consistency)":
            w_sent, w_lex, w_subj, w_typo = 0.55, 0.15, 0.15, 0.15
        else:
            w_sent, w_lex, w_subj, w_typo = 0.25, 0.25, 0.25, 0.25

        st.subheader("Heuristic Weights")
        disabled_weights = preset_mode != "Custom"
        if disabled_weights:
            st.info(f"Using **{preset_mode}** profile weights. Select 'Custom' above to manually adjust sliders.")

        weight_sentiment = st.slider(
            "Sentiment vs Rating Weight",
            min_value=0.0,
            max_value=1.0,
            value=w_sent,
            step=0.05,
            disabled=disabled_weights,
            help="Penalty weight for reviews where star rating contradicts text sentiment.",
        )
        weight_lexical = st.slider(
            "Lexical Diversity Weight",
            min_value=0.0,
            max_value=1.0,
            value=w_lex,
            step=0.05,
            disabled=disabled_weights,
            help="Penalty weight for low Type-Token Ratio and repetitive copy-paste phrases.",
        )
        weight_subjectivity = st.slider(
            "Subjectivity & Fluff Weight",
            min_value=0.0,
            max_value=1.0,
            value=w_subj,
            step=0.05,
            disabled=disabled_weights,
            help="Penalty weight for extreme subjective claims lacking concrete nouns/specs.",
        )
        weight_typography = st.slider(
            "Typographical Anomaly Weight",
            min_value=0.0,
            max_value=1.0,
            value=w_typo,
            step=0.05,
            disabled=disabled_weights,
            help="Penalty weight for excessive ALL-CAPS words and punctuation marks (!!!, ???).",
        )

        with st.expander("🛠️ Advanced Thresholds", expanded=False):
            ttr_threshold = st.slider(
                "Min Expected TTR Threshold",
                min_value=0.40,
                max_value=0.85,
                value=0.65,
                step=0.05,
                help="Type-Token Ratio below this triggers repetition penalties.",
            )
            max_caps = st.slider(
                "Max Allowed ALL-CAPS Ratio",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.05,
                help="Percentage of words in all-caps allowed before penalty kicks in.",
            )

        st.divider()
        st.markdown(
            """
            **Engine Architecture**:
            - Deterministic heuristic pipeline (`TextBlob` + Regex).
            - Sub-millisecond latency; zero GPU requirement.
            - Evaluates semantic coherence, vocabulary richness, emotional fluff, and typographic structure.
            """
        )

    # -------------------------------------------------------------------------
    # MAIN INPUT AREA
    # -------------------------------------------------------------------------
    weights = {
        "sentiment": weight_sentiment,
        "lexical": weight_lexical,
        "subjectivity": weight_subjectivity,
        "typography": weight_typography,
    }

    col_input, col_meta = st.columns([3, 2])

    with col_meta:
        st.subheader("1. Select or Load Example")
        sample_choice = st.selectbox(
            "Quick-load realistic test samples:",
            options=list(SAMPLE_REVIEWS.keys()),
            index=0,
        )
        sample_data = SAMPLE_REVIEWS[sample_choice]
        if sample_data["desc"]:
            st.caption(f"ℹ️ **Scenario**: {sample_data['desc']}")

    preset_text = sample_data["text"]
    preset_rating = sample_data["rating"]

    with col_input:
        st.subheader("2. Review Details")
        star_rating = st.slider(
            "Customer Star Rating:",
            min_value=1,
            max_value=5,
            value=preset_rating,
            format="%d ⭐",
            help="The declared rating submitted with this review.",
        )

    review_text = st.text_area(
        "Paste Review Text:",
        value=preset_text,
        height=140,
        placeholder="Type or paste an e-commerce product review here...",
        help="Review text to analyze with heuristic NLP algorithms.",
    )

    if not review_text.strip():
        st.warning("⚠️ Enter some review text above or select a preset sample to view the Trust Score.")
        return

    # -------------------------------------------------------------------------
    # EXECUTE NLP HEURISTICS
    # -------------------------------------------------------------------------
    sentiment_res = analyze_sentiment_divergence(review_text, star_rating)
    lexical_res = analyze_lexical_diversity(review_text, ttr_threshold=ttr_threshold)
    subjectivity_res = analyze_subjectivity(review_text)
    typo_res = analyze_typographical_anomalies(review_text, max_caps_ratio=max_caps)

    composite = compute_composite_trust_score(
        sentiment_res, lexical_res, subjectivity_res, typo_res, weights
    )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # OUTPUT: DYNAMIC TRUST SCORE HERO CARD
    # -------------------------------------------------------------------------
    st.subheader("3. Trust & Credibility Assessment")

    trust_score = composite["trust_score"]
    verdict = composite["verdict"]
    verdict_class = composite["verdict_class"]
    badge_icon = composite["badge_icon"]
    explanation = composite["explanation"]

    st.markdown(
        f"""
        <div class="trust-hero-card {verdict_class}">
            <div style="font-size: 0.95rem; color: #64748b; font-weight: 600; text-transform: uppercase;">
                Overall Review Trust Score
            </div>
            <div class="trust-score-num">
                {trust_score:.1f}%
            </div>
            <div class="trust-verdict-title">
                {badge_icon} {verdict}
            </div>
            <div style="margin-top: 10px; font-size: 1.05rem; max-width: 720px; margin-left: auto; margin-right: auto;">
                {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Streamlit native progress bar
    st.progress(
        value=int(trust_score),
        text=f"Trust Level: {trust_score:.1f}% (Penalty Deduction: {composite['weighted_penalty']:.1f}%)",
    )

    # -------------------------------------------------------------------------
    # METRICS AT A GLANCE (COLUMNS)
    # -------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Sentiment Divergence",
            value=f"{sentiment_res['divergence']:.2f}",
            delta=f"-{sentiment_res['penalty']:.0f} pts" if sentiment_res["penalty"] > 0 else "0 pts",
            delta_color="inverse",
            help="Normalized divergence between star rating and text polarity.",
        )
    with m2:
        st.metric(
            label="Lexical Diversity (TTR)",
            value=f"{lexical_res['ttr']:.2f}",
            delta=f"-{lexical_res['penalty']:.0f} pts" if lexical_res["penalty"] > 0 else "0 pts",
            delta_color="inverse",
            help="Unique tokens / Total tokens. Lower scores indicate repetitive or bot vocabulary.",
        )
    with m3:
        st.metric(
            label="Subjectivity Score",
            value=f"{subjectivity_res['subjectivity']:.2f}",
            delta=f"-{subjectivity_res['penalty']:.0f} pts" if subjectivity_res["penalty"] > 0 else "0 pts",
            delta_color="inverse",
            help="Subjectivity ratio (0=Objective, 1=Pure Fluff/Hype).",
        )
    with m4:
        st.metric(
            label="Typo / Caps Anomaly",
            value=f"{typo_res['caps_ratio']*100:.1f}% Caps",
            delta=f"-{typo_res['penalty']:.0f} pts" if typo_res["penalty"] > 0 else "0 pts",
            delta_color="inverse",
            help="Ratio of multi-letter words in all-caps + punctuation runs.",
        )

    # -------------------------------------------------------------------------
    # DIAGNOSTICS: INTERACTIVE DATAFRAME BREAKDOWN
    # -------------------------------------------------------------------------
    st.subheader("4. Heuristic Diagnostics Breakdown")

    norm_w = composite["normalized_weights"]
    diagnostics_data = [
        {
            "Heuristic": "Sentiment vs Rating Divergence",
            "Status": sentiment_res["flag"],
            "Raw Metric": f"Expected: {sentiment_res['expected_polarity']:+.2f} | Polarity: {sentiment_res['actual_polarity']:+.2f}",
            "Penalty (0-100)": sentiment_res["penalty"],
            "Configured Weight": f"{norm_w['sentiment']*100:.1f}%",
            "Trust Impact": f"-{sentiment_res['penalty'] * norm_w['sentiment']:.1f}%",
            "Finding": sentiment_res["summary"],
        },
        {
            "Heuristic": "Lexical Diversity (TTR)",
            "Status": lexical_res["flag"],
            "Raw Metric": f"TTR: {lexical_res['ttr']:.2f} ({lexical_res['unique_tokens']}/{lexical_res['total_tokens']} words)",
            "Penalty (0-100)": lexical_res["penalty"],
            "Configured Weight": f"{norm_w['lexical']*100:.1f}%",
            "Trust Impact": f"-{lexical_res['penalty'] * norm_w['lexical']:.1f}%",
            "Finding": lexical_res["summary"],
        },
        {
            "Heuristic": "Subjectivity & Fluff",
            "Status": subjectivity_res["flag"],
            "Raw Metric": f"Subj: {subjectivity_res['subjectivity']:.2f} | {subjectivity_res['superlatives_count']} Superlatives vs {subjectivity_res['specs_count']} Specs",
            "Penalty (0-100)": subjectivity_res["penalty"],
            "Configured Weight": f"{norm_w['subjectivity']*100:.1f}%",
            "Trust Impact": f"-{subjectivity_res['penalty'] * norm_w['subjectivity']:.1f}%",
            "Finding": subjectivity_res["summary"],
        },
        {
            "Heuristic": "Typographical Anomalies",
            "Status": typo_res["flag"],
            "Raw Metric": f"Caps: {typo_res['caps_ratio']*100:.1f}% | Punct Runs: {typo_res['excessive_punct_runs']}",
            "Penalty (0-100)": typo_res["penalty"],
            "Configured Weight": f"{norm_w['typography']*100:.1f}%",
            "Trust Impact": f"-{typo_res['penalty'] * norm_w['typography']:.1f}%",
            "Finding": typo_res["summary"],
        },
    ]

    df_diagnostics = pd.DataFrame(diagnostics_data)

    def style_status(val: str) -> str:
        if val == "Flagged":
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"
        elif val == "Warning":
            return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
        return "background-color: #dcfce7; color: #166534; font-weight: bold;"

    styled_df = df_diagnostics.style.map(style_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # EXPANDABLE DEEP-DIVE DETAILS
    # -------------------------------------------------------------------------
    with st.expander("🔍 Deep-Dive NLP Technical Findings", expanded=False):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### 1. Sentiment & Rating Divergence")
            st.write(
                f"- **Declared Rating**: `{star_rating} ⭐` (Normalized: `{sentiment_res['expected_polarity']:+.2f}`)"
            )
            st.write(f"- **Text Sentiment Polarity**: `{sentiment_res['actual_polarity']:+.2f}` (Scale: -1.0 to +1.0)")
            st.write(f"- **Normalized Divergence**: `{sentiment_res['divergence']:.2f}` (Scale: 0.0 to 1.0)")
            st.write(f"- **Raw Heuristic Penalty**: `{sentiment_res['penalty']} / 100`")

            st.markdown("#### 2. Lexical Repetition & Token Metrics")
            st.write(f"- **Total Multi-letter Tokens**: `{lexical_res['total_tokens']}`")
            st.write(f"- **Unique Tokens**: `{lexical_res['unique_tokens']}`")
            st.write(f"- **Type-Token Ratio (TTR)**: `{lexical_res['ttr']:.3f}`")
            st.write(f"- **Root TTR (Length-stabilized)**: `{lexical_res['root_ttr']:.2f}`")
            st.write(f"- **Identical Trigram Repetitions**: `{lexical_res['repeated_trigrams']}`")

        with c2:
            st.markdown("#### 3. Subjectivity & Feature Specificity")
            st.write(f"- **Subjectivity Level**: `{subjectivity_res['subjectivity']:.2f}` (0 = Objective, 1 = Pure Opinion)")
            st.write(f"- **Detected Superlative Phrases**: `{subjectivity_res['superlatives_count']}`")
            if subjectivity_res["found_superlatives"]:
                st.caption(f"Matches: {', '.join(f'`{w}`' for w in subjectivity_res['found_superlatives'][:6])}")
            st.write(f"- **Concrete Specifications / Nouns**: `{subjectivity_res['specs_count']}`")

            st.markdown("#### 4. Typographical & Punctuation Patterns")
            st.write(f"- **ALL-CAPS Words**: `{typo_res['caps_words_count']}` ({typo_res['caps_ratio']*100:.1f}%)")
            if typo_res.get("caps_words"):
                st.caption(f"Words: {', '.join(f'`{w}`' for w in typo_res['caps_words'][:8])}")
            st.write(f"- **Excessive Punctuation Runs (!!/??)**: `{typo_res['excessive_punct_runs']}`")
            st.write(f"- **Character Elongations (soooo)**: `{typo_res['elongated_words']}`")


if __name__ == "__main__":
    main()
