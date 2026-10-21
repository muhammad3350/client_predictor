import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataprocessing import build_preprocessor, load_data, split_xy

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ---------- 1) Data Loading & Split ----------
df = load_data()
X, y = split_xy(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.2, max_iter=1000, class_weight="balanced", random_state=42
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=25, weights="distance", p=1
    ),
    "Naive Bayes": GaussianNB(
        var_smoothing=0.01
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=20, min_samples_leaf=15, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=7, min_samples_split=10, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=42
    ),
    "SVM": SVC(
        C=0.5, kernel="rbf", probability=True, class_weight="balanced", random_state=42
    ),
    "Voting Ensemble": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.2, max_iter=1000, class_weight="balanced", random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=42)),
        ],
        voting="soft"
    ),
}

# ---------- 3) train + evaluate ----------
results = []
trained = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", clf),
    ])
    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)
    probabilities = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })
    trained[name] = pipe
    print(f"done: {name}")

# ---------- 4) compare ----------
results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n", results_df.round(3).to_string(index=False))

best_name = results_df.loc[0, "Model"]
print(f"\nBest model (by F1): {best_name}")

# ---------- 5) save ----------
results_df.to_csv(MODELS_DIR / "results.csv", index=False)
joblib.dump(trained[best_name], MODELS_DIR / "churn_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)