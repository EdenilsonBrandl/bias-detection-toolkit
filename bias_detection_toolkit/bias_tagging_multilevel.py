"""
Module: bias_tagging_multilevel_nlp.py

Performs 3-level tagging of input responses or data points using advanced NLP models
(spaCy, transformers) to infer psychological, cognitive, epistemic, and behavioral signals.

Author: Edenilson Brandl
"""

import spacy
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import re

class MultilevelBiasTaggerNLP:
    def __init__(self):
        # Load NLP pipelines
        self.nlp = spacy.load("en_core_web_trf")

        # Load emotion classifier
        self.emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

        # Load bias/political classifier
        self.bias_pipe = pipeline("text-classification", model="unitary/toxic-bert")

    def tag_level_1(self, text: str) -> List[str]:
        tags = []
        emotion_results = self.emotion_pipe(text)
        top_emotions = [r['label'] for r in emotion_results[0] if r['score'] > 0.5]
        if any(e in top_emotions for e in ['anger', 'fear', 'sadness', 'disgust']):
            tags.append("emotional_influence")

        bias_result = self.bias_pipe(text)
        if any(r['label'] == 'toxic' and r['score'] > 0.5 for r in bias_result):
            tags.append("cognitive_bias")

        if re.search(r"confused|misremember|foggy|vague|unsure", text, re.IGNORECASE):
            tags.append("information_noise_or_telephone_effect")

        if re.search(r"brain|amygdala|neuro|frontal|dopamine|impulse|neurological", text, re.IGNORECASE):
            tags.append("neurological_insufficiency")

        return tags

    def tag_level_2(self, text: str) -> List[str]:
        tags = []
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_stop]

        if any(l in lemmas for l in ["science", "experiment", "empirical", "research"]):
            tags.append("scientific_basis")
        if any(l in lemmas for l in ["experience", "memory", "feel", "believe"]):
            tags.append("personal_experience")
        if any(l in lemmas for l in ["punish", "reward", "training", "conditioning"]):
            tags.append("behaviorist_conditioning")
        if any(l in lemmas for l in ["brain", "dopamine", "cortex"]):
            tags.append("neurological_denotation")
        if any(l in lemmas for l in ["symbol", "imply", "metaphor"]):
            tags.append("connotative_thought")
        if any(l in lemmas for l in ["literal", "concrete", "exact"]):
            tags.append("denotative_thought")

        return tags

    def tag_level_3(self, text: str, lvl1: List[str], lvl2: List[str]) -> List[str]:
        tags = []
        doc = self.nlp(text.lower())
        if "cognitive_bias" in lvl1 and "personal_experience" in lvl2:
            tags.append("subpersonality_pattern")
        if "behaviorist_conditioning" in lvl2 and "emotional_influence" in lvl1:
            tags.append("social_engineering_influence")
        if "connotative_thought" in lvl2 and "information_noise_or_telephone_effect" in lvl1:
            tags.append("conceptual_dispersal")
        if re.search(r"astrology|chakras|energy_field|detox|homeopathy|quantum_healing", text.lower()):
            tags.append("pseudoscience_influence")
        return tags

    def analyze_text(self, text: str) -> Dict[str, List[str]]:
        level1 = self.tag_level_1(text)
        level2 = self.tag_level_2(text)
        level3 = self.tag_level_3(text, level1, level2)
        return {
            "Level 1 Tags": level1,
            "Level 2 Tags": level2,
            "Level 3 Tags": level3
        }

# Example usage:
if __name__ == "__main__":
    sample_text = (
        "I feel like the government is rigged, but maybe it's just my bias."
        "My brain struggles to keep up with politics."
        "Some say astrology and energy healing explain my emotions."
    )
    tagger = MultilevelBiasTaggerNLP()
    result = tagger.analyze_text(sample_text)

    import pprint
    pprint.pprint(result)
