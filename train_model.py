from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).parent / "final_dataset.csv"
ARTIFACT = Path(__file__).parent / "cheat_detection_model.joblib"


def main():
    data = pd.read_csv(DATA)
    if "cheat" not in data or "uuid" not in data:
        raise ValueError("Dataset must contain 'cheat' and 'uuid' columns.")
    y = data["cheat"].astype(int)
    X = data.drop(columns=["cheat", "uuid"])
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
            X, y, groups=data["uuid"]
        )
    )
    model = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    scores = model.predict_proba(X.iloc[test_idx])[:, 1]
    print(f"UUID-grouped accuracy: {accuracy_score(y.iloc[test_idx], predictions):.4f}")
    print(f"UUID-grouped ROC AUC: {roc_auc_score(y.iloc[test_idx], scores):.4f}")
    print("Confusion matrix [actual rows, predicted columns]:")
    print(confusion_matrix(y.iloc[test_idx], predictions))
    print(classification_report(y.iloc[test_idx], predictions, zero_division=0))
    model.fit(X, y)
    joblib.dump({"model": model, "features": list(X.columns)}, ARTIFACT)
    print(f"Saved trained model to {ARTIFACT}")


if __name__ == "__main__":
    main()
