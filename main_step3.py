"""main.py
API FastAPI d'analyse d'avis clients.

Utilise :
- clean_text et detect_sentiment_simple depuis data_processing.py
- generer_reponse depuis generate_response.py

Endpoint principal :
POST /analyse
Body :
{
    "texte": "Le produit est nul"
}

Réponse :
{
    "texte": "...",
    "sentiment": "negatif",
    "reponse": "...",
    "email_client": "client023@example.com"
}
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import random

from generate_response import generer_reponse
from data_processing import clean_text, detect_sentiment_simple


app = FastAPI(title="API Analyse et Réponse Client")


class AnalyseRequest(BaseModel):
    texte: str


class AnalyseResponse(BaseModel):
    texte: str
    sentiment: Literal["positif", "negatif"]
    reponse: str
    email_client: str


def generate_fake_email() -> str:
    """Génère un email artificiel de type clientXYZ@example.com"""
    return f"client{random.randint(1, 99999):03d}@example.com"


@app.post("/analyse", response_model=AnalyseResponse)
def analyse_avis(req: AnalyseRequest) -> AnalyseResponse:
    """Analyse un avis : nettoie le texte, prédit le sentiment, génère une réponse."""
    # Nettoyer le texte (principalement utile si on veut logger ou améliorer plus tard)
    texte_nettoye = clean_text(req.texte)

    # Déterminer le sentiment via notre règle simple
    sentiment = detect_sentiment_simple(texte_nettoye)

    # Générer la réponse : seulement vraiment nécessaire si avis négatif,
    # mais pour simplifier on l'appelle dans tous les cas
    if sentiment == "negatif":
        reponse = generer_reponse(req.texte)
    else:
        reponse = "Merci pour votre avis positif et votre confiance !"

    email = generate_fake_email()

    return AnalyseResponse(
        texte=req.texte,
        sentiment=sentiment,
        reponse=reponse,
        email_client=email,
    )


# Pour lancer en local :
#   uvicorn main:app --reload
