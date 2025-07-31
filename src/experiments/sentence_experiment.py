# src/experiments/sentence_experiment.py
import sys
from pathlib import Path
import os
import joblib
import logging
sys.path.append("C:/Users/aledi/review-classifier-NLP/src")  # Asegura que src esté en el path
from data import load_data
from embeddings import SentenceTransformerEmbedding
from models import LogisticModel, XGBModel, RandomForestModel
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from utils import set_mongo_connection, setup_logging

# Configurar logging
setup_logging()

def run_sentence_experiment():
    # Conexión manual a MongoDB
    client = None
    try:
        client = set_mongo_connection()

        # Cargar datos
        logging.info("Loading data from MongoDB...")

        # Access database and collection
        db = client["reviews_db"]
        collection = db["processed__reviews"]

        logging.info("Extracting texts and califications...")

        # Extract texts and califications using list comprehension with filters
        docs = list(collection.find({
            "raw_text": {"$exists": True},
            "calification": {"$exists": True}
        }))

        texts = [doc["raw_text"] for doc in docs]
        califications = [doc["calification"] for doc in docs]

        logging.info(f"Loaded {len(texts)} documents.")

        # Convert califications to binary labels
        labels = []
        for calif in califications:
            try:
                calif_num = float(calif)  # Convert string to float
                if calif_num < 5:
                    labels.append(0)  # Negativo
                elif calif_num > 5:
                    labels.append(1)  # Positivo
            except (ValueError, TypeError):
                logging.warning(f"Invalid calification value skipped: {calif}")
                continue

        # texts, labels = load_data(client, "processed__reviews")

        # Convertir a strings si son listas de palabras
        #texts_str = [" ".join(t) if isinstance(t, list) else t for t in texts]

        # Dividir datos en entrenamiento y prueba
        X_train_str, X_test_str, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Instancia de embedding
        emb_instance = SentenceTransformerEmbedding()

        EMB_PATH = Path(__file__).resolve().parent.parent / "embeddings" / "sentence_model.joblib"
        if EMB_PATH.exists():
            logging.info(f"Loading existing sentence embedding from {EMB_PATH}")
            emb_instance = emb_instance.load(EMB_PATH)
        else:
            logging.info("Training sentence embeddings...")
            emb_instance.save(EMB_PATH)
            logging.info(f"Saved sentence embedding to {EMB_PATH}")

        # Transformar datos
        X_train_emb = emb_instance.transform(X_train_str)
        X_test_emb = emb_instance.transform(X_test_str)

        # Instancias de modelos
        models = {
            "logistic": LogisticModel(C=1.0),
            "xgboost": XGBModel(n_estimators=100, learning_rate=0.1),
            "random_forest": RandomForestModel(n_estimators=100, max_depth=10)
        }

        MODEL_FOLDER = Path(__file__).resolve().parent.parent / "models"
        MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

        # Ejecutar experimentos
        for model_name, model_instance in models.items():
            MODEL_PATH = MODEL_FOLDER / f"{model_name}_sentence_model.joblib"
            if MODEL_PATH.exists():
                logging.info(f"Loading existing {model_name} model with sentence from {MODEL_PATH}")
                model_instance.model = joblib.load(MODEL_PATH)
            else:
                logging.info(f"Training {model_name} with sentence...")
                model_instance.train(X_train_emb, y_train)

            accuracy, report = model_instance.evaluate(X_test_emb, y_test)
            logging.info(f"Model: {model_name}, Accuracy: {accuracy:.2f}")
            logging.info(f"Report:\n{report}")

            joblib.dump(model_instance.model, MODEL_PATH)
            logging.info(f"Saved {model_name} model to {MODEL_PATH}")

    except Exception as e:
        logging.error(f"Error during experiment: {e}")
    finally:
        if client is not None:
            client.close()
            logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    run_sentence_experiment()