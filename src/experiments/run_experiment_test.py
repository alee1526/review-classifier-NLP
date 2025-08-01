import sys
from pathlib import Path
import os
import joblib
import logging
import pandas as pd
from sklearn.metrics import classification_report
sys.path.append("C:/Users/aledi/review-classifier-NLP/src")  # Asegura que src esté en el path
from data import load_data
from embeddings import TfidfEmbedding, Word2VecEmbedding, SentenceTransformerEmbedding
from models import LogisticModel, XGBModel, RandomForestModel
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from utils import set_mongo_connection, setup_logging


# Configurar logging
setup_logging()

def run_experiment_test():
    # Conexión manual a MongoDB
    client = None
    # Crear DataFrame para almacenar resultados
    results_df = pd.DataFrame(columns=["Embedding", "Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    
    try:
        client = set_mongo_connection()

        # Cargar datos
        logging.info("Loading data from MongoDB...")
        texts, labels = load_data(client, "test_processed__reviews")

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
        RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
        RESULTS_FOLDER.mkdir(exist_ok=True)  # Crear carpeta si no existe
        RESULTS_PATH = RESULTS_FOLDER / "experiment_results.csv"

        # Ejecutar experimentos
        for emb_name, emb_instance in embeddings.items():
            EMB_PATH = EMB_FOLDER / f"{emb_name}_model.joblib"

            if EMB_PATH.exists():
                logging.info(f"Loading existing {emb_name} embedding from {EMB_PATH}")
                emb_instance = emb_instance.load(EMB_PATH)
                X_test_emb = emb_instance.transform([" ".join(t) for t in texts])
                logging.info(f"Embedding {emb_name} loaded and transformed.")
            else:
                logging.info(f"{emb_name} is not trained yet.")

            for model_name, model_instance in models.items():
                MODEL_PATH = MODEL_FOLDER / f"{model_name}_{emb_name}_model.joblib"

                if MODEL_PATH.exists():
                    logging.info(f"Loading existing {model_name} model with {emb_name} from {MODEL_PATH}")
                    model_instance.model = joblib.load(MODEL_PATH)
                    accuracy, report = model_instance.evaluate(X_test_emb, labels)
                    logging.info(f"Embedding: {emb_name}, Model: {model_name}, Accuracy: {accuracy:.2f}")
                    logging.info(f"Report:\n{report}")

                    # Extraer métricas del reporte (asumiendo que report es un dict de classification_report)
                    report_dict = classification_report(labels, model_instance.model.predict(X_test_emb), output_dict=True)
                    precision = report_dict["weighted avg"]["precision"]
                    recall = report_dict["weighted avg"]["recall"]
                    f1 = report_dict["weighted avg"]["f1-score"]

                    # Agregar resultados al DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([{
                        "Embedding": emb_name,
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1
                    }])], ignore_index=True)
                else:
                    logging.info(f"{model_name} with {emb_name} is not trained yet.")

        # Guardar DataFrame en CSV
        results_df.to_csv(RESULTS_PATH, index=False)
        logging.info(f"Results saved to {RESULTS_PATH}")
        logging.info("Experiment completed successfully.")

    except Exception as e:
        logging.error(f"Error during experiment: {e}")
    finally:
        if client is not None:
            client.close()
            logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    run_experiment_test()