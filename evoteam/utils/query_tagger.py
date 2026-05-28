"""
Query tagger for semantic classification of attack queries.

Tags queries by content category for cache lookup and plan matching.
Uses keyword-based classification (no external dependencies).
"""

from typing import List, Dict, Set

# Predefined tag categories with keyword mappings
TAG_KEYWORDS: Dict[str, Set[str]] = {
    "violence": {
        "kill", "murder", "bomb", "weapon", "gun", "shoot", "attack",
        "violent", "violence", "harm", "injure", "hurt", "destroy",
        "explosive", "war", "combat", "fight", "stab", "torture",
    },
    "drugs": {
        "drug", "cocaine", "heroin", "meth", "opioid", "narcotic",
        "synthetic", "synthesize", "manufacture", "cannabis", "lsd",
        "fentanyl", "pill", "prescription", "pharmacy",
    },
    "hate_speech": {
        "hate", "racist", "racism", "discriminate", "supremacist",
        "xenophobic", "bigot", "anti-semitic", "islamophobic",
        "homophobic", "transphobic", "hate speech",
    },
    "misinformation": {
        "fake", "misinformation", "disinformation", "propaganda",
        "conspiracy", "hoax", "false claim", "mislead", "deceive",
        "fabricate", "fabricated", "rumor",
    },
    "self_harm": {
        "suicide", "self-harm", "self harm", "cut myself", "kill myself",
        "ending my life", "want to die", "self injury", "self-injury",
    },
    "illegal_activity": {
        "illegal", "crime", "criminal", "steal", "theft", "robbery",
        "fraud", "scam", "hack", "hacking", "phishing", "counterfeit",
        "forgery", "money laundering", "trafficking", "smuggle",
    },
    "chemical_biological": {
        "chemical weapon", "biological weapon", "nerve agent", "mustard gas",
        "anthrax", "ricin", "sarin", "toxic gas", "bioweapon", "pathogen",
        "virus weaponize", "toxin",
    },
    "cyber_attack": {
        "malware", "ransomware", "virus", "worm", "trojan", "botnet",
        "ddos", "denial of service", "exploit", "zero-day", "backdoor",
        "keylogger", "rootkit", "spyware",
    },
    "privacy_violation": {
        "dox", "doxxing", "surveillance", "stalk", "spy on",
        "private information", "personal data", "identity theft",
        "social engineer", "impersonate",
    },
    "sexual_content": {
        "sexual", "porn", "nude", "naked", "explicit", "obscene",
        "child abuse", "csam", "rape", "sexual assault",
    },
}


class QueryTagger:
    """Tags attack queries with semantic categories for cache lookup."""

    def __init__(self, custom_keywords: Dict[str, Set[str]] = None):
        self.keywords = dict(TAG_KEYWORDS)
        if custom_keywords:
            for tag, words in custom_keywords.items():
                if tag in self.keywords:
                    self.keywords[tag].update(words)
                else:
                    self.keywords[tag] = set(words)

    def tag(self, query: str) -> List[str]:
        """Extract all matching tags for a query. Returns list of tag strings."""
        query_lower = query.lower()
        matched_tags = []

        for tag, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_tags.append(tag)
                    break  # One keyword match is enough per tag

        return matched_tags if matched_tags else ["general"]

    def primary_tag(self, query: str) -> str:
        """Get the primary (most specific) tag for a query."""
        tags = self.tag(query)
        return tags[0]  # First match is considered primary

    def tag_with_difficulty(self, query: str) -> Dict:
        """Tag query and estimate difficulty."""
        tags = self.tag(query)
        query_lower = query.lower()

        # Difficulty heuristics based on query complexity
        word_count = len(query.split())
        has_specifics = any(w in query_lower for w in [
            "step by step", "detailed", "specific", "exact", "precise"
        ])

        if word_count > 50:
            difficulty = "hard"
        elif word_count > 20 or has_specifics:
            difficulty = "medium"
        else:
            difficulty = "easy"

        return {
            "tags": tags,
            "primary_tag": tags[0],
            "difficulty": difficulty,
            "word_count": word_count,
        }


# Predefined tag distance matrix (semantic similarity between tags)
TAG_SIMILARITY = {
    ("violence", "illegal_activity"): 0.7,
    ("violence", "chemical_biological"): 0.6,
    ("drugs", "illegal_activity"): 0.7,
    ("drugs", "chemical_biological"): 0.5,
    ("hate_speech", "violence"): 0.5,
    ("cyber_attack", "illegal_activity"): 0.8,
    ("cyber_attack", "privacy_violation"): 0.7,
    ("misinformation", "illegal_activity"): 0.4,
    ("chemical_biological", "violence"): 0.6,
}


def get_nearest_tag(target_tag: str, available_tags: Set[str]) -> str:
    """Find the nearest tag by semantic similarity."""
    if target_tag in available_tags:
        return target_tag

    best_tag = "general"
    best_similarity = 0.0

    for avail in available_tags:
        similarity = TAG_SIMILARITY.get((target_tag, avail), 0.0)
        similarity = max(similarity, TAG_SIMILARITY.get((avail, target_tag), 0.0))
        if similarity > best_similarity:
            best_similarity = similarity
            best_tag = avail

    return best_tag