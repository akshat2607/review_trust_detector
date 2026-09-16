"""
Automated unit tests for NLP Heuristics Engine in app.py
"""
import unittest
from app import (
    analyze_sentiment_divergence,
    analyze_lexical_diversity,
    analyze_subjectivity,
    analyze_typographical_anomalies,
    compute_composite_trust_score,
    SAMPLE_REVIEWS,
)

class TestNLPHeuristics(unittest.TestCase):

    def test_sentiment_divergence_aligned(self):
        # 5 stars with positive text
        res = analyze_sentiment_divergence("This is an excellent and wonderful speaker with great sound!", 5)
        self.assertLess(res["divergence"], 0.38)
        self.assertEqual(res["penalty"], 0.0)
        self.assertEqual(res["flag"], "Pass")

    def test_sentiment_divergence_severe_mismatch(self):
        # 5 stars with scathing negative text
        res = analyze_sentiment_divergence("Horrible defective trash! It broke instantly, terrible seller and total scam.", 5)
        self.assertGreater(res["divergence"], 0.5)
        self.assertGreaterEqual(res["penalty"], 80.0)
        self.assertEqual(res["flag"], "Flagged")

    def test_sentiment_divergence_inverted_mismatch(self):
        # 1 star with glowing praise
        res = analyze_sentiment_divergence("Absolutely love this, fantastic quality and brilliant performance!", 1)
        self.assertGreater(res["divergence"], 0.6)
        self.assertGreaterEqual(res["penalty"], 80.0)
        self.assertEqual(res["flag"], "Flagged")

    def test_lexical_diversity_repetitive_bot(self):
        # Repetitive keyword spam
        spam = "Best speaker ever buy speaker now best speaker buy speaker speaker speaker."
        res = analyze_lexical_diversity(spam, ttr_threshold=0.65)
        self.assertLess(res["ttr"], 0.55)
        self.assertGreater(res["penalty"], 30.0)

    def test_lexical_diversity_healthy(self):
        # Natural rich vocabulary
        text = "The packaging was minimal yet protective, and the setup took barely five minutes on my laptop."
        res = analyze_lexical_diversity(text, ttr_threshold=0.65)
        self.assertGreaterEqual(res["ttr"], 0.70)
        self.assertEqual(res["penalty"], 0.0)
        self.assertEqual(res["flag"], "Pass")

    def test_subjectivity_hype_fluff(self):
        # Superlatives without specs
        fluff = "BEST EVER LIFE CHANGING MIRACLE 100% RECOMMEND HOLY GRAIL UNBELIEVABLE!"
        res = analyze_subjectivity(fluff)
        self.assertGreaterEqual(res["superlatives_count"], 4)
        self.assertEqual(res["specs_count"], 0)
        self.assertGreaterEqual(res["penalty"], 50.0)

    def test_subjectivity_grounded_specs(self):
        # Realistic specs and details
        review = "The 5000mAh battery lasted 14 hours during continuous video playback, and the aluminum frame weighs 180g."
        res = analyze_subjectivity(review)
        self.assertGreaterEqual(res["specs_count"], 3)
        self.assertLess(res["penalty"], 20.0)

    def test_typographical_anomalies_caps_and_punct(self):
        # All caps and punctuation runs
        screaming = "OMG BUY THIS NOW THIS IS SO AMAZINGGGGG!!!!!!! BEST ITEM EVER??? YES!!!!"
        res = analyze_typographical_anomalies(screaming)
        self.assertGreater(res["caps_ratio"], 0.40)
        self.assertGreater(res["excessive_punct_runs"], 1)
        self.assertGreaterEqual(res["penalty"], 40.0)

    def test_composite_trust_score_sample_bot(self):
        sample = SAMPLE_REVIEWS["🤖 Bot Copy-Paste Spam (Repetitive Keywords & Low TTR)"]
        s_res = analyze_sentiment_divergence(sample["text"], sample["rating"])
        l_res = analyze_lexical_diversity(sample["text"])
        sub_res = analyze_subjectivity(sample["text"])
        t_res = analyze_typographical_anomalies(sample["text"])
        weights = {"sentiment": 0.35, "lexical": 0.25, "subjectivity": 0.20, "typography": 0.20}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        # Should be classified as High Risk (< 50%)
        self.assertLess(comp["trust_score"], 50.0)
        self.assertEqual(comp["verdict_class"], "fake")

    def test_composite_trust_score_sample_glitch(self):
        sample = SAMPLE_REVIEWS["⚡ Inverted Rating Glitch (5-Star Rating with Scathing Negative Text)"]
        s_res = analyze_sentiment_divergence(sample["text"], sample["rating"])
        l_res = analyze_lexical_diversity(sample["text"])
        sub_res = analyze_subjectivity(sample["text"])
        t_res = analyze_typographical_anomalies(sample["text"])
        weights = {"sentiment": 0.35, "lexical": 0.25, "subjectivity": 0.20, "typography": 0.20}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        # Should be classified as High Risk (< 50%)
        self.assertLess(comp["trust_score"], 50.0)
        self.assertEqual(comp["verdict_class"], "fake")

    def test_composite_trust_score_sample_fluff(self):
        sample = SAMPLE_REVIEWS["📢 Paid Hype Fluff (Superlatives, Excessive Caps & Exclamations, No Specs)"]
        s_res = analyze_sentiment_divergence(sample["text"], sample["rating"])
        l_res = analyze_lexical_diversity(sample["text"])
        sub_res = analyze_subjectivity(sample["text"])
        t_res = analyze_typographical_anomalies(sample["text"])
        weights = {"sentiment": 0.35, "lexical": 0.25, "subjectivity": 0.20, "typography": 0.20}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        # Should be classified as High Risk (< 50%)
        self.assertLess(comp["trust_score"], 50.0)
        self.assertEqual(comp["verdict_class"], "fake")

    def test_composite_trust_score_sample_authentic(self):
        sample = SAMPLE_REVIEWS["✅ Genuine Authentic Review (Balanced, Concrete Details & Specs)"]
        s_res = analyze_sentiment_divergence(sample["text"], sample["rating"])
        l_res = analyze_lexical_diversity(sample["text"])
        sub_res = analyze_subjectivity(sample["text"])
        t_res = analyze_typographical_anomalies(sample["text"])
        weights = {"sentiment": 0.35, "lexical": 0.25, "subjectivity": 0.20, "typography": 0.20}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        # Should be authentic with trust score >= 80%
        self.assertGreaterEqual(comp["trust_score"], 80.0)
        self.assertEqual(comp["verdict_class"], "authentic")

    def test_composite_trust_score_sample_critical(self):
        sample = SAMPLE_REVIEWS["📉 Genuine Critical Review (1-Star, Detailed Factual Defect)"]
        s_res = analyze_sentiment_divergence(sample["text"], sample["rating"])
        l_res = analyze_lexical_diversity(sample["text"])
        sub_res = analyze_subjectivity(sample["text"])
        t_res = analyze_typographical_anomalies(sample["text"])
        weights = {"sentiment": 0.35, "lexical": 0.25, "subjectivity": 0.20, "typography": 0.20}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        # Should be authentic with trust score >= 80%
        self.assertGreaterEqual(comp["trust_score"], 80.0)
        self.assertEqual(comp["verdict_class"], "authentic")

    def test_edge_cases_empty_text(self):
        s_res = analyze_sentiment_divergence("", 5)
        l_res = analyze_lexical_diversity("")
        sub_res = analyze_subjectivity("")
        t_res = analyze_typographical_anomalies("")
        weights = {"sentiment": 0.25, "lexical": 0.25, "subjectivity": 0.25, "typography": 0.25}
        comp = compute_composite_trust_score(s_res, l_res, sub_res, t_res, weights)
        self.assertEqual(comp["trust_score"], 100.0)

if __name__ == "__main__":
    unittest.main()
