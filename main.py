# Machine Learning Project - Vertebral Column Classification
# Dataset: dataset_spine.csv
# Problem: classify patients as Normal or Abnormal using biomechanical measurements.

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, RocCurveDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

# ===============================
# 1. Configurare reproductibila
# ===============================
SEED = 42
TARGET = "Class_att"
TEST_SIZE = 0.20
DATA_PATH = "./data/dataset_spine.csv"
POSITIVE_LABEL = "Abnormal"


def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def get_metrics(y_true, y_pred, y_score=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_score is not None:
        result["roc_auc"] = roc_auc_score(y_true, y_score)
    return result


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    row = {"model": name}
    row.update(get_metrics(y_test, y_pred, y_score))
    return row, y_pred, y_score, model


def print_table(title, rows):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# ===============================
# 2. Incarcare dataset
# ===============================
df = clean_columns(pd.read_csv(DATA_PATH))
TARGET = TARGET.strip().replace(" ", "_")
X = df.drop(columns=[TARGET])
y = (df[TARGET] == POSITIVE_LABEL).astype(int)  # 1 = Abnormal, 0 = Normal
feature_names = list(X.columns)

print("Dimensiune dataset:", df.shape)
print("Coloane:", list(df.columns))
print("Tipuri date:\n", df.dtypes)
print("\nDistributia claselor count:\n", df[TARGET].value_counts())
print("\nDistributia claselor %:\n", (df[TARGET].value_counts(normalize=True) * 100).round(2))
print("\nMissing values count:\n", df.isna().sum())
print("\nMissing values %:\n", (df.isna().mean() * 100).round(2))
print("\nStatistici descriptive:\n", X.describe().round(3))

# Outlieri prin regula IQR
outlier_rows = []
for col in feature_names:
    q1, q3 = X[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = ((X[col] < low) | (X[col] > high)).sum()
    outlier_rows.append({"feature": col, "lower": low, "upper": high, "outliers": count, "pct": count / len(X) * 100})
print_table("Outlieri IQR", outlier_rows)

# ===============================
# 3. Split train/test
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# ===============================
# 4. Scenarii preprocesare si data leakage
# ===============================
scenario_rows = []
for model_name, base_model in [
    ("kNN", KNeighborsClassifier(n_neighbors=5)),
    ("Decision Tree", DecisionTreeClassifier(random_state=SEED, max_depth=4)),
]:
    # fara transformari
    model = base_model.__class__(**base_model.get_params())
    row, *_ = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
    row.update({"scenario": "fara_transformari", "scaler": "none"})
    scenario_rows.append(row)

    # corect: scaler fit doar pe train, aplicat pe test prin Pipeline
    for scaler_name, scaler in [
        ("StandardScaler", StandardScaler()),
        ("MinMaxScaler", MinMaxScaler()),
        ("RobustScaler", RobustScaler()),
    ]:
        model = Pipeline([
            ("scaler", scaler),
            ("model", base_model.__class__(**base_model.get_params()))
        ])
        row, *_ = evaluate_model(model_name, model, X_train, X_test, y_train, y_test)
        row.update({"scenario": "corect_dupa_split", "scaler": scaler_name})
        scenario_rows.append(row)

    # gresit: scaler fit pe tot datasetul inainte de split
    scaler = StandardScaler()
    X_scaled_wrong = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
    Xtr_w, Xte_w, ytr_w, yte_w = train_test_split(
        X_scaled_wrong, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    model = base_model.__class__(**base_model.get_params())
    row, *_ = evaluate_model(model_name, model, Xtr_w, Xte_w, ytr_w, yte_w)
    row.update({"scenario": "gresit_inainte_de_split", "scaler": "StandardScaler"})
    scenario_rows.append(row)

print_table("Comparatie scenarii preprocesare", scenario_rows)

# ===============================
# 5. Stabilitate: split-uri si CV
# ===============================
split_rows = []
for test_size in [0.40, 0.30, 0.20]:
    for seed in [1, 42, 99]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        for name, model in [
            ("kNN", Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))])),
            ("Decision Tree", DecisionTreeClassifier(random_state=seed, max_depth=4)),
        ]:
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            split_rows.append({
                "model": name, "test_size": test_size, "seed": seed,
                "accuracy": accuracy_score(yte, yp),
                "f1_macro": f1_score(yte, yp, average="macro")
            })
print_table("Stabilitate split train/test", split_rows)

cv_rows = []
cv_models = {
    "kNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED, max_depth=4),
    "Gaussian NB": GaussianNB(),
    "MLP": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(30,), alpha=0.001, max_iter=5000, random_state=SEED))]),
}
for folds in [5, 10]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for name, model in cv_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
        cv_rows.append({"model": name, "folds": folds, "f1_macro_mean": scores.mean(), "f1_macro_std": scores.std()})
print_table("Cross-validation 5 vs 10 fold", cv_rows)

# ===============================
# 6. Resampling pentru dezechilibru
# ===============================
resampling_rows = []
for name in ["kNN", "Decision Tree"]:
    for method in ["none", "RandomOverSampler", "SMOTE"]:
        if name == "kNN":
            steps = [("scaler", StandardScaler())]
            if method == "RandomOverSampler":
                steps.append(("ros", RandomOverSampler(random_state=SEED)))
            elif method == "SMOTE":
                steps.append(("smote", SMOTE(random_state=SEED)))
            steps.append(("model", KNeighborsClassifier(n_neighbors=5)))
            model = ImbPipeline(steps)
        else:
            steps = []
            if method == "RandomOverSampler":
                steps.append(("ros", RandomOverSampler(random_state=SEED)))
            elif method == "SMOTE":
                steps.append(("smote", SMOTE(random_state=SEED)))
            steps.append(("model", DecisionTreeClassifier(random_state=SEED, max_depth=4)))
            model = ImbPipeline(steps)

        row, *_ = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        row["resampling"] = method
        resampling_rows.append(row)
print_table("Comparatie resampling", resampling_rows)

# ===============================
# 7. Modele finale
# ===============================
models = {
    "kNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5, metric="euclidean"))]),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED, criterion="gini", max_depth=4),
    "Gaussian NB": GaussianNB(),
    "MLP": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(30,), alpha=0.001, max_iter=5000, random_state=SEED))]),
}

final_rows = []
fitted = {}
for name, model in models.items():
    row, y_pred, y_score, trained = evaluate_model(name, model, X_train, X_test, y_train, y_test)
    final_rows.append(row)
    fitted[name] = (trained, y_pred, y_score)
    print("\nClassification report:", name)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

print_table("Tabel comparativ final", final_rows)

# ===============================
# 8. Studiu parametru
# ===============================
param_rows = []
for k in [1, 3, 5, 7, 9, 11, 15]:
    model = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=k))])
    row, *_ = evaluate_model("kNN", model, X_train, X_test, y_train, y_test)
    row.update({"param": "n_neighbors", "value": k})
    param_rows.append(row)

for depth in [1, 2, 3, 4, 5, None]:
    model = DecisionTreeClassifier(random_state=SEED, max_depth=depth)
    row, *_ = evaluate_model("Decision Tree", model, X_train, X_test, y_train, y_test)
    row.update({"param": "max_depth", "value": depth})
    param_rows.append(row)
print_table("Studiu parametru", param_rows)

# ===============================
# 9. GridSearchCV pentru 2 modele
# ===============================
knn_grid = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    {
        "scaler": [StandardScaler(), MinMaxScaler(), RobustScaler()],
        "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "model__metric": ["euclidean", "manhattan"],
    },
    cv=5,
    scoring="f1_macro",
)
knn_grid.fit(X_train, y_train)

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=SEED),
    {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, None],
        "min_samples_leaf": [1, 2, 5, 10],
    },
    cv=5,
    scoring="f1_macro",
)
dt_grid.fit(X_train, y_train)

for name, grid in [("kNN", knn_grid), ("Decision Tree", dt_grid)]:
    print("\nGridSearch", name)
    print("Best params:", grid.best_params_)
    print("Best CV f1_macro:", round(grid.best_score_, 4))
    yp = grid.predict(X_test)
    ys = grid.predict_proba(X_test)[:, 1]
    print(get_metrics(y_test, yp, ys))

# ===============================
# 10. Interpretari modele
# ===============================
# kNN: distanta, k, scalare
knn_detail_rows = []
for metric in ["euclidean", "manhattan"]:
    for k in [3, 5, 9]:
        for scaler_name, scaler in [("none", None), ("StandardScaler", StandardScaler())]:
            steps = []
            if scaler is not None:
                steps.append(("scaler", scaler))
            steps.append(("model", KNeighborsClassifier(n_neighbors=k, metric=metric)))
            model = Pipeline(steps)
            row, *_ = evaluate_model("kNN", model, X_train, X_test, y_train, y_test)
            row.update({"distance": metric, "k": k, "scaler": scaler_name})
            knn_detail_rows.append(row)
print_table("kNN: distanta, k si scalare", knn_detail_rows)

# Decision Tree: reguli + feature importance
tree = models["Decision Tree"]
tree.fit(X_train, y_train)
print("\nFeature importance Decision Tree:")
print(pd.Series(tree.feature_importances_, index=feature_names).sort_values(ascending=False).round(4))
print("\nReguli Decision Tree:")
print(export_text(tree, feature_names=feature_names, max_depth=3))

# GaussianNB: parametri estimati
nb = GaussianNB().fit(X_train, y_train)
print("\nGaussianNB theta_ medii pe clase, rand 0=Normal, rand 1=Abnormal")
print(pd.DataFrame(nb.theta_, columns=feature_names).round(3))
print("\nGaussianNB var_ variante pe clase, rand 0=Normal, rand 1=Abnormal")
print(pd.DataFrame(nb.var_, columns=feature_names).round(3))

# MLP: loss curve
mlp = models["MLP"].fit(X_train, y_train)
loss = mlp.named_steps["model"].loss_curve_
print("\nMLP loss: iteratii=", len(loss), "first=", round(loss[0], 4), "last=", round(loss[-1], 4))

# ===============================
# 11. Analiza erorilor
# ===============================
best_name = "Decision Tree"
best_model, y_pred, y_score = fitted[best_name]
errors = X_test.copy()
errors["true"] = y_test.map({0: "Normal", 1: "Abnormal"})
errors["pred"] = pd.Series(y_pred, index=X_test.index).map({0: "Normal", 1: "Abnormal"})
errors = errors[errors["true"] != errors["pred"]]
print("\nExemple gresite pentru", best_name)
print(errors.head(5).round(3))

# ===============================
# 12. Grafice obligatorii
# ===============================
os.makedirs("output", exist_ok=True)

for name, (model, y_pred, y_score) in fitted.items():
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Abnormal"])
    disp.plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"output/confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_score, name=name)
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(f"output/roc_curve_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

print("\nGrafice salvate ca PNG in folderul 'output'.")
