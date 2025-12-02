# Dockerfile pour l'API FastAPI (Qwen2.5-3B)

# 1. Image de base Python
FROM python:3.10-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code de l'application
COPY . .

# 5. Exposer le port (optionnel, mais pratique en local)
EXPOSE 8000

# 6. Commande de lancement (compatible local / Azure) :
#    - écoute sur 0.0.0.0
#    - port lu dans la variable d'environnement PORT (sinon 8000)
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
