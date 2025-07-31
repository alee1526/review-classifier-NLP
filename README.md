# Review Classifier NLP Project

This repository contains a natural language processing (NLP) project designed to classify movie reviews as positive or negative using various embedding techniques and machine learning models. The project utilizes the IMDb dataset, processes it with MongoDB Atlas, and experiments with different models to evaluate their performance.

## Project Overview

The project follows a structured workflow to download data, store it in a MongoDB database, preprocess the text, and train machine learning models using different embedding techniques.

### Workflow
1. **Data Download and Storage:**
   - The IMDb dataset was downloaded and uploaded to a MongoDB Atlas database using the `mongodb_data_upload.py` script.
   - The database contains a collection with 12,500 positive reviews and 12,500 negative reviews, totaling 25,000 documents.

2. **Data Preprocessing:**
   - The raw text data was processed to create a new attribute called `processed_text` for each document.
   - Processing involved removing common words (stopwords) using SpaCy and lemmatizing the remaining words to normalize the text.
   - The processed data is stored alongside the raw text in the MongoDB collection.

3. **Model Development and Management:**
   - The project implements a class-based architecture to manage embeddings and machine learning models.
   - **Embedding Classes:** Custom classes (`TfidfEmbedding`, `Word2VecEmbedding`, and `SentenceTransformerEmbedding`) handle the generation of text embeddings, with methods for fitting, transforming, saving, and loading embeddings.
   - **Model Classes:** Custom classes (`LogisticModel`, `XGBModel`, and `RandomForestModel`) encapsulate the machine learning models, providing methods for training, evaluation, and saving/loading model instances.
   - These classes are organized in the `src/embeddings/` and `src/models/` directories, promoting modularity and reusability.

4. **Experiments:**
   - Experiments were conducted using two scripts:
     - `run_experiment.py`: Runs experiments with `tfidf` and `word2vec` embeddings on all three models.
     - `sentence_experiment.py`: Runs experiments with `sentence_transformer` embeddings (using raw text) on the same models.

5. **Results:**
   - The models were evaluated on a test set of 5,000 reviews (2,515 negative, 2,485 positive).
   - Performance metrics are reported below in tables for each embedding technique and model combination.

## Results

### TF-IDF Embedding
| Model            | Accuracy | Class 0 (Precision/Recall/F1) | Class 1 (Precision/Recall/F1) |
|-------------------|----------|------------------------------|------------------------------|
| Logistic Regression | 0.88     | 0.90 / 0.85 / 0.87          | 0.86 / 0.90 / 0.88          |
| XGBoost           | 0.81     | 0.85 / 0.76 / 0.80          | 0.78 / 0.86 / 0.82          |
| Random Forest     | 0.82     | 0.85 / 0.79 / 0.82          | 0.80 / 0.86 / 0.83          |

### Word2Vec Embedding
| Model            | Accuracy | Class 0 (Precision/Recall/F1) | Class 1 (Precision/Recall/F1) |
|-------------------|----------|------------------------------|------------------------------|
| Logistic Regression | 0.87     | 0.88 / 0.84 / 0.86          | 0.85 / 0.89 / 0.87          |
| XGBoost           | 0.86     | 0.87 / 0.85 / 0.86          | 0.85 / 0.87 / 0.86          |
| Random Forest     | 0.83     | 0.85 / 0.81 / 0.83          | 0.82 / 0.85 / 0.83          |

### Sentence Transformer Embedding (Raw Text)
| Model            | Accuracy | Class 0 (Precision/Recall/F1) | Class 1 (Precision/Recall/F1) |
|-------------------|----------|------------------------------|------------------------------|
| Logistic Regression | 0.81     | 0.82 / 0.81 / 0.81          | 0.81 / 0.82 / 0.81          |
| XGBoost           | 0.79     | 0.79 / 0.78 / 0.79          | 0.78 / 0.80 / 0.79          |
| Random Forest     | 0.76     | 0.77 / 0.73 / 0.75          | 0.74 / 0.78 / 0.76          |

## Setup Instructions

### Prerequisites
- Python 3.11
- Conda (recommended for environment management)

### Environment Setup
1. Clone the repository:
   ```
   git clone https://github.com/your-username/review-classifier-nlp.git
   cd review-classifier-nlp
   ```
2. Create the Conda environment using the provided `environment.yml`:
   ```
   conda env create -f environment.yml -n nlp-reviews
   ```
   Or update an existing environment:
   ```
   conda env update -f environment.yml --prune -n nlp-reviews
   ```
3. Activate the environment:
   ```
   conda activate nlp-reviews
   ```

### Data Preparation
1. Download the IMDb dataset and place it in the `data/raw/aclImdb_v1/aclImdb` directory.
2. Upload the data to MongoDB Atlas using:
   ```
   python mongodb_data_upload.py
   ```
   This creates a database with a collection containing 12,500 positive and 12,500 negative reviews.

### Running Experiments
1. For TF-IDF and Word2Vec experiments:
   ```
   python src/experiments/run_experiment.py
   ```
   - Models and embeddings are saved in `src/models/` and `src/embeddings/`.
2. For Sentence Transformer experiments (using raw text):
   ```
   python src/experiments/sentence_experiment.py
   ```
   - Models are saved in `src/models/`.

### Notes
- The `data` and `notebooks` directories are excluded from version control via `.gitignore`.
- Processed text is stored in the `processed_text` attribute in MongoDB after preprocessing with SpaCy.
- Model performance varies by embedding technique, with TF-IDF and Word2Vec generally outperforming Sentence Transformer on this dataset, possibly due to preprocessing effects on semantic context.

## Contributing
Feel free to fork this repository, submit issues, or propose enhancements. Contributions to improve preprocessing, model tuning, or experiment scripts are welcome!

## License
[Add your license here, e.g., MIT License] - Specify if applicable.