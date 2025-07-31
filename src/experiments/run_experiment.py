# src/experiments/run_experiment.py
import sys
from pathlib import Path
import os
import joblib
import logging
sys.path.append("C:/Users/aledi/review-classifier-NLP/src")  # Asegura que src esté en el path
from data import load_data
from embeddings import TfidfEmbedding, Word2VecEmbedding, SentenceTransformerEmbedding
from models import LogisticModel, XGBModel, RandomForestModel
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from utils import set_mongo_connection, setup_logging


# Configurar logging
setup_logging()

def run_experiment():
    # Conexión manual a MongoDB
    client = None
    try:
        client = set_mongo_connection()

        # Cargar datos
        logging.info("Loading data from MongoDB...")
        texts, labels = load_data(client, "processed__reviews")

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

        # Instancias de embeddings
        embeddings = {
            "tfidf": TfidfEmbedding(),
            "word2vec": Word2VecEmbedding(vector_size=200, window=10, min_count=5)
        }

        # Instancias de modelos
        models = {
            "logistic": LogisticModel(C=1.0),
            "xgboost": XGBModel(n_estimators=100, learning_rate=0.1),
            "random_forest": RandomForestModel(n_estimators=100, max_depth=10)
        }

        EMB_FOLDER = Path(__file__).resolve().parent.parent / "embeddings"  
        MODEL_FOLDER = Path(__file__).resolve().parent.parent / "models"

        # Ejecutar experimentos
        for emb_name, emb_instance in embeddings.items():
            EMB_PATH = EMB_FOLDER / f"{emb_name}_model.joblib"
            if EMB_PATH.exists():
                logging.info(f"Loading existing {emb_name} embedding from {EMB_PATH}")
                emb_instance = emb_instance.load(EMB_PATH)
            else:
                logging.info(f"Training {emb_name} embeddings...")
                X_train_str = [" ".join(t) for t in X_train]
                emb_instance.fit(X_train_str)

            emb_instance.save(EMB_PATH)
            logging.info(f"Saved {emb_name} embedding to {EMB_PATH}")
            X_test_emb = emb_instance.transform([" ".join(t) for t in X_test])

            for model_name, model_instance in models.items():
                MODEL_PATH = MODEL_FOLDER / f"{model_name}_{emb_name}_model.joblib"
                if MODEL_PATH.exists():
                    logging.info(f"Loading existing {model_name} model with {emb_name} from {MODEL_PATH}")
                    model_instance.model = joblib.load(MODEL_PATH)
                else:
                    logging.info(f"Training {model_name} with {emb_name}...")
                    X_train_emb = emb_instance.transform([" ".join(t) for t in X_train])
                    model_instance.train(X_train_emb, y_train)

                accuracy, report = model_instance.evaluate(X_test_emb, y_test)
                logging.info(f"Embedding: {emb_name}, Model: {model_name}, Accuracy: {accuracy:.2f}")
                logging.info(f"Report:\n{report}")

                joblib.dump(model_instance, MODEL_PATH)
                logging.info(f"Saved {model_name} model to {MODEL_PATH}")

    except Exception as e:
        logging.error(f"Error during experiment: {e}")
    finally:
        if client is not None:
            client.close()
            logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    run_experiment()