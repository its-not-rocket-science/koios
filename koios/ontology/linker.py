"""
Entity linker for mapping surface text spans to OWL ontology classes.

Supports exact string matching and basic fuzzy matching.
"""

import re
from difflib import get_close_matches


class EntityLinker:
    """
    EntityLinker class for linking text spans to ontology classes.
    This class provides methods for exact and fuzzy matching of ontology class names
    to text spans. It can be used to link entities in a text to their corresponding
    ontology classes.

    Attributes:
        class_names (list of str): List of ontology class names to match against.
        label_to_id (dict): Mapping from class names to unique IDs.
        Methods:

            exact_match(text): Finds all exact matches of ontology class names in the input text.
            fuzzy_match(mention, cutoff=0.85): Finds the closest ontology class to a mention using fuzzy string match.
            link_text(text, method="exact", cutoff=0.8, return_tokens=False): Link all spans in the text to ontology classes.
            Returns a list of dictionaries containing the span, mention, and class name.
            If return_tokens=True, returns token-level {token_id, entity_id}.
    """

    def __init__(self, class_names):
        """
        Args:
            class_names (list of str): Ontology class names to match against.
        """
        self.class_names = sorted(set(class_names), key=len, reverse=True)
        self.label_to_id = {name: i + 1 for i,
                            name in enumerate(self.class_names)}

    def exact_match(self, text):
        """
        Finds all exact matches of ontology class names in the input text.

        Args:
            text (str): Input sentence or document.

        Returns:
            list of (span, class_name) tuples.
        """
        matches = []
        for cname in self.class_names:
            for m in re.finditer(rf"\b{re.escape(cname)}\b", text, re.IGNORECASE):
                matches.append(((m.start(), m.end()), cname))
        return matches

    def fuzzy_match(self, mention, cutoff=0.85):
        """
        Finds the closest ontology class to a mention using fuzzy string match.

        Args:
            mention (str): Span text to match.
            cutoff (float): Similarity threshold.

        Returns:
            str or None
        """
        candidates = get_close_matches(
            mention, self.class_names, n=1, cutoff=cutoff)
        return candidates[0] if candidates else None

    def link_text(self, text: str, method: str = "exact", cutoff: float = 0.8, return_tokens: bool = False):
        """
        Link all spans in the text to ontology classes.

        Args:
            text (str): The input text.
            method (str): "exact" or "fuzzy"
            cutoff (float): Similarity threshold for fuzzy match
            return_tokens (bool): If True, returns token-level {token_id, entity_id}

        Returns:
            list of dicts: If return_tokens=False → [{"span": (start, end), "mention": "...", "class": "..."}]
                        If return_tokens=True  → [{"token_id": int, "text": str, "entity_id": int}]
        """
        if return_tokens:
            tokens = text.split()
            linked = []
            for idx, tok in enumerate(tokens):
                tok_clean = tok.strip(".,!?;:").capitalize()
                ent_id = 0
                if method == "exact":
                    ent_id = self.label_to_id.get(tok_clean, 0)
                elif method == "fuzzy":
                    best_score = cutoff
                    for cname in self.class_names:
                        score = 1 - (self._levenshtein(tok_clean,
                                     cname) / max(len(tok_clean), len(cname)))
                        if score >= best_score:
                            ent_id = self.label_to_id[cname]
                            best_score = score
                linked.append({
                    "token_id": idx,
                    "text": tok,
                    "entity_id": ent_id
                })
                if not linked:
                    linked.append({
                        "token_id": 0,
                        "text": "<PAD>",
                        "entity_id": 0
                    })
            return linked

        results = []
        if method == "exact":
            for cname in self.class_names:
                if cname in text:
                    idx = text.find(cname)
                    results.append({
                        "span": (idx, idx + len(cname)),
                        "mention": cname,
                        "class": cname
                    })
        elif method == "fuzzy":
            for word in text.split():
                best_score = cutoff
                best_match = None
                for cname in self.class_names:
                    score = 1 - (self._levenshtein(word, cname) / max(len(word), len(cname)))
                    if score >= best_score:
                        best_match = cname
                        best_score = score
                if best_match:
                    idx = text.find(word)
                    results.append({
                        "span": (idx, idx + len(word)),
                        "mention": word,
                        "class": best_match
                    })
        return results

    def _levenshtein(self, a, b):
        """
        Compute the Levenshtein distance between two strings.
        Args:
            a (str): First string.
            b (str): Second string.
        Returns:
            int: Levenshtein distance.
        """
        if len(a) < len(b):
            return self._levenshtein(b, a)  # pylint: disable=arguments-out-of-order
        if len(b) == 0:
            return len(a)
        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def is_valid_class(self, class_id):
        """
        Checks if the given class ID corresponds to a known ontology class.

        Args:
            class_id (int): The class index predicted by the model.

        Returns:
            bool: True if the class_id maps to a known ontology class, False otherwise.
        """
        print(f"Checking if class_id {class_id} is in {self.label_to_id.values()}")
        return class_id in self.label_to_id.values()
