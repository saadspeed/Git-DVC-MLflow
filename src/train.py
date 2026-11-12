
import os
import yaml
import json
import joblib
import mlflow
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# 1. Définir le chemin racine du projet
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================
# 2. Charger les paramètres
# =====================================

with open(BASE_DIR / "params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]

# =====================================
# 3. Charger le dataset Iris
# =====================================

df = pd.read_csv(BASE_DIR / "data" / "raw" / "iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=y
)

# =====================================
# 4. Configuration MLflow
# =====================================

mlflow.set_tracking_uri(
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
)

mlflow.set_experiment("TP_Iris_Classification")

# =====================================
# 5. Définir les modèles
# =====================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}

# =====================================
# 6. Créer les dossiers
# =====================================

os.makedirs(BASE_DIR / "models", exist_ok=True)
os.makedirs(BASE_DIR / "reports", exist_ok=True)

# =====================================
# 7. Entraîner et évaluer les modèles
# =====================================

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calcul des métriques
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),

            "precision": precision_score(
                y_test, predictions, average="weighted"
            ),

            "recall": recall_score(
                y_test, predictions, average="weighted"
            ),

            "f1_score": f1_score(
                y_test, predictions, average="weighted"
            )
        }

        # Enregistrer les paramètres
        mlflow.log_params(params)
        mlflow.log_param("model", name)

        # Enregistrer les métriques
        mlflow.log_metrics(metrics)

        # Sauvegarder le modèle
        model_path = BASE_DIR / "models" / f"{name}.pkl"

        joblib.dump(model, model_path)

        mlflow.log_artifact(str(model_path))

        # Sauvegarder les métriques en JSON
        report_path = BASE_DIR / "reports" / f"{name}_metrics.json"

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n==============================")
        print("MODELE :", name)
        print("==============================")

        for metric, value in metrics.items():
            print(f"{metric} : {value:.4f}")

print("\nTous les modèles ont été entraînés avec succès !")
