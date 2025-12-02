"""data_processing.py
Pré-traitement des avis :
- nettoyage de texte
- détection simple du sentiment (positif / negatif)
"""

import re
import string

# Stopwords français simples
FRENCH_STOPWORDS = {
    "a", "à", "ai", "aie", "aient", "aies", "ait", "alors", "as", "au", "aucun", "aura",
    "aurai", "auraient", "aurais", "aurait", "auve", "avec", "avez", "aviez", "avions",
    "avoir", "avons", "bon", "car", "ce", "cela", "ces", "cet", "cette", "ceux", "chaque",
    "comme", "d", "dans", "de", "des", "du", "elle", "en", "encore", "est", "et", "eu",
    "fait", "faites", "fois", "ici", "il", "ils", "je", "la", "le", "les", "leur", "lui",
    "mais", "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou", "par",
    "pas", "pour", "plus", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
    "ta", "te", "tes", "toi", "ton", "toujours", "tout", "tous", "très", "tu",
    "un", "une", "vos", "votre", "vous", "y"
}

PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

# Mots indicateurs d'avis négatif (règle très simple)
NEGATIVE_KEYWORDS = {"nul", "mauvais", "horrible", "cassé", "casse", "déçu", "pire", "remboursement"}


def clean_text(text: str) -> str:
    """Nettoyage simple du texte pour l'analyse."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in FRENCH_STOPWORDS]
    return " ".join(tokens)


def detect_sentiment_simple(texte: str) -> str:
    """Détection très simple du sentiment à partir de quelques mots-clés."""
    clean = clean_text(texte)
    tokens = set(clean.split())
    if any(mot in tokens for mot in NEGATIVE_KEYWORDS):
        return "negatif"
    return "positif"


if __name__ == "__main__":
    ex1 = "Le produit est nul, je suis très déçu."
    ex2 = "Super qualité, je recommande ce produit !"
    print(ex1, "->", detect_sentiment_simple(ex1))
    print(ex2, "->", detect_sentiment_simple(ex2))
