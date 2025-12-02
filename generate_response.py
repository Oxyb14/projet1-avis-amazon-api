"""generate_response.py
Chargement du modèle Qwen2.5-3B-Instruct et génération de réponses
pour les avis négatifs.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"[INFO] Chargement du modèle de réponse : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")

# Gestion du token de padding
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
    device=device,
)


def build_reply_prompt(review_text: str) -> str:
    """Construit le prompt de réponse type service client Amazon."""
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


if __name__ == "__main__":
    exemple = "Le produit est arrivé cassé, je suis très déçue."
    print("Avis :", exemple)
    print("Réponse générée :", generer_reponse(exemple))
