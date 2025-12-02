from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import random
import re
import string

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ============================================================
# 1. Nettoyage de texte + détection de sentiment simple
# ============================================================

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

# quelques mots indicateurs d'avis négatif (règle simple)
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
    """ 
    Détection très simple du sentiment :
    - si au moins un mot 'négatif' est présent -> 'negatif'
    - sinon -> 'positif'
    """
    clean = clean_text(texte)
    tokens = set(clean.split())
    if any(mot in tokens for mot in NEGATIVE_KEYWORDS):
        return "negatif"
    return "positif"


# ============================================================
# 2. Modèle Qwen2.5-3B pour générer la réponse
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"[INFO] Chargement du modèle : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

# gestion du token de padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print("[INFO] GPU détecté : utilisation de cuda:0")
else:
    print("[INFO] Pas de GPU détecté : utilisation du CPU (plus lent)")

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)


def build_reply_prompt(review_text: str) -> str:
    """Prompt de réponse type service client Amazon."""
    lignes = [
        "Tu es un agent du service client Amazon.",
        "Tu dois répondre en français à un avis client NEGATIF.",
        "",
        "Consignes pour ta réponse :",
        "- rester poli et professionnel,",
        "- reconnaître le problème,",
        "- proposer une solution ou un contact,",
        "- utiliser un ton empathique.",
        "",
        "Avis du client :",
        review_text,
        "",
        "Réponse du service client :",
    ]
    return "\n".join(lignes)


def generer_reponse(review_text: str,
                    max_new_tokens: int = 80,
                    temperature: float = 0.5,
                    top_p: float = 0.9) -> str:
    """Génère une réponse IA au texte d'avis fourni."""
    prompt = build_reply_prompt(review_text)
    out = gen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=1,
    )[0]["generated_text"]

    split_token = "Réponse du service client :"
    if split_token in out:
        return out.split(split_token, 1)[1].strip()
    return out.strip()


# ============================================================
# 3. API FastAPI
# ============================================================

app = FastAPI(
    title="API Analyse d'avis Amazon",
    description=(
        "Envoie un texte, prédit le sentiment, génère une réponse "
        "et renvoie le JSON complet."
    ),
    version="1.0.0",
)


class AnalyseRequest(BaseModel):
    texte: str


class AnalyseResponse(BaseModel):
    texte: str
    sentiment: Literal["positif", "negatif"]
    reponse: str
    email_client: str


def generate_fake_email() -> str:
    """Génère un email artificiel (clientXXX@example.com)."""
    return f"client{random.randint(1, 99999):03d}@example.com"


@app.get("/")
async def root():
    return {"message": "API d'analyse d'avis Amazon - voir /docs pour tester l'endpoint /analyse."}


@app.post("/analyse", response_model=AnalyseResponse)
async def analyse_avis(payload: AnalyseRequest) -> AnalyseResponse:
    """ 
    Endpoint principal :
    - entrée : payload.texte
    - sortie : JSON {texte, sentiment, reponse, email_client}
    """
    # 1. Sentiment
    sentiment = detect_sentiment_simple(payload.texte)

    # 2. Réponse IA
    if sentiment == "negatif":
        reponse = generer_reponse(payload.texte)
    else:
        reponse = "Merci pour votre avis positif et votre confiance !"

    # 3. Email artificiel
    email = generate_fake_email()

    return AnalyseResponse(
        texte=payload.texte,
        sentiment=sentiment,
        reponse=reponse,
        email_client=email,
    )


# Pour lancer en local :
#   pip install -r requirements.txt
#   uvicorn main:app --reload
