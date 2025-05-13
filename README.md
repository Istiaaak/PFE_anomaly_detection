# Détection d’anomalies pour une chaîne de production



## Présentation
Ce dépôt propose une chaîne complète de détection d’anomalies :

| Bloc | Techno | Fichier principal |
|------|--------|-------------------|
| Producteur d’images | **Kafka Producer** | `producer.py` |
| Message bus | **Apache Kafka** | Docker image |
| Service d’inférence | **FastAPI** (+ PyTorch PatchCore) | `api_app.py` |
| Tableau de bord | **Streamlit** (+ Plotly) | `dashboard.py` |
| Logs structurés |  JSON | `logger.py` |

Le modèle **PatchCore** apprend uniquement sur les images *good* du jeu
[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) et
signale toute déviation à l’aide d’un score ≥ seuil (80ᵉ percentile par
défaut).

---

## Mode rapide : Docker
```
# 1. clonage
git clone https://github.com/<votre-org>/patchcore-project.git
cd patchcore-project

# 2. build + run (Kafka + API + Dashboard + Producer)
docker compose up -d --build

# 3. vérifier
docker compose ps              # tous les conteneurs "running"
xdg-open http://localhost:8501 # ouvre le dashboard
```
## Mode manuel : sans Docker
```
# 0) venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Kafka local
export KAFKA_HOME=$HOME/kafka_2.13-3.7.2
$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties &
$KAFKA_HOME/bin/kafka-server-start.sh     $KAFKA_HOME/config/server.properties &

# 2) API
uvicorn api_app:app --host 0.0.0.0 --port 8000

# 3) Dashboard
streamlit run dashboard.py --server.port 8501

```


## Utilisation de l’API

| Action                 | Appel                                                                 |
| ---------------------- | --------------------------------------------------------------------- |
| **Build memory bank**           | `POST /build` — JSON `{"cls":"bottle","backbone_key":"WideResNet50"}` |
| **Prédire fichier**    | `POST /predict` (`multipart/form-data` : `file=@image.png`)           |
| **Prédire flux Kafka** | `GET /stream_predict`                                                 |

Exemple :

```
curl -X POST http://localhost:8000/build/ \
     -H "Content-Type: application/json" \
     -d '{"cls":"bottle","backbone_key":"WideResNet50"}'
```
```
curl -X POST http://localhost:8000/predict/ \
     -F file=@datasets/bottle/test/good/000.png
```
