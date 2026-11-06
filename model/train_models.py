import argparse
import importlib.util
from importlib import metadata as package_metadata
import json
from pathlib import Path
import platform
import sys

SEED = 42                  
TEST_FRACTION = 0.20        
TARGET = "booked_with_card_7d"          
CLASS_NAMES = ["No booking (0)", "Booked (1)"]
DATASET_SOURCE = "Synthetic data generated with seed 42"
FEATURE_COLUMNS = ['is_credit_card',
 'is_debit_card',
 'is_prepaid_card',
 'card_tenure_months',
 'transactions_last_30d',
 'avg_transaction_inr_30d',
 'days_since_last_purchase',
 'offer_views_last_90d',
 'offer_bookings_last_90d',
 'travel_txns_last_90d',
 'dining_txns_last_90d',
 'historical_email_open_rate',
 'offered_discount_pct',
 'minimum_spend_inr',
 'offer_validity_days',
 'shown_on_weekend',
 'shown_in_app']
INSTALL_COMMAND = "%pip install numpy pandas scikit-learn matplotlib joblib streamlit"
CORE_PACKAGES = {
    "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn",
    "matplotlib": "matplotlib", "joblib": "joblib",
}
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def package_version(distribution):
    """Return an installed package version without importing the package."""
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return None


def heading(number, title):
    """Print a visible separator so the output is easy to read."""
    print("\n" + "=" * 72)
    print(f"STEP {number}: {title}")
    print("=" * 72)




def validate_test_csv(frame, feature_names):
    """Validate schema, numeric values and optional binary labels before inference."""
    if frame.empty:
        raise ValueError("The CSV is empty. Upload at least one test row.")
    if len(frame) > 5000:
        raise ValueError("Upload test data only, with at most 5,000 rows.")
    missing = sorted(set(feature_names) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_names) - {TARGET})
    if missing or extra:
        raise ValueError(f"Column mismatch. Missing: {missing}; unexpected: {extra}. Use test_data.csv.")
    # Explicit feature order prevents a shuffled CSV from changing predictions.
    X = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be numeric and contain no blanks, NaN or infinity.")
    y = None
    if TARGET in frame.columns:
        y = pd.to_numeric(frame[TARGET], errors="raise")
        if y.isna().any() or not y.isin([0, 1]).all():
            raise ValueError("booked_with_card_7d must contain only 0=no booking or 1=booked, with no blanks.")
        y = y.astype(int)
    return X, y

def step_1_environment(state):
    heading(1, "Check Python and packages")
    print("Python:", platform.python_version())
    missing = []
    for module, distribution in CORE_PACKAGES.items():
        available = importlib.util.find_spec(module) is not None
        print(f"{distribution:15} : {package_version(distribution) or 'NOT INSTALLED'}")
        if not available:
            missing.append(distribution)
    print(f"{'streamlit':15} : {package_version('streamlit') or 'not installed; needed later'}")
    if missing:
        raise RuntimeError("Missing: " + ", ".join(missing) + "\nNotebook cell: " + INSTALL_COMMAND)
    print("Training packages are available. Continue to Step 2.")


def load_ml_imports():
    """Import third-party packages AFTER Step 1 has checked availability."""
    # global lets the functions below use these imported names.
    global np, pd, joblib, plt, train_test_split
    global StratifiedKFold, cross_val_score, Pipeline, StandardScaler
    global LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier
    global GaussianNB, SVC, accuracy_score, roc_auc_score, precision_score
    global recall_score, f1_score, matthews_corrcoef, classification_report
    global confusion_matrix, ConfusionMatrixDisplay
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")  # Save charts even when the lab has no desktop display.
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score, recall_score,
        f1_score, matthews_corrcoef, classification_report,
        confusion_matrix, ConfusionMatrixDisplay,
    )

def step_2_dataset(state):
    heading(2, "Load and understand the dataset")
    csv_path = state["csv"]
    if not csv_path.is_file():
        raise ValueError(f"CSV not found: {csv_path}. Keep card_campaign_bookings.csv beside this script, or use --csv PATH.")
    # read_csv reads the first row as column names and later rows as samples.
    frame = pd.read_csv(csv_path)
    # The booking outcome is already encoded as 0 or 1.
    if TARGET not in frame.columns:
        raise ValueError("Training CSV needs booked_with_card_7d: 0=no booking, 1=booked.")
    # Check the expected feature names and put them in a fixed order.
    expected_features = FEATURE_COLUMNS
    X, y = validate_test_csv(frame, expected_features)
    if set(y.unique()) != {0, 1} or y.value_counts().min() < 10:
        raise ValueError("Both target classes need at least 10 rows for this teaching workflow.")
    print("Read CSV:", csv_path)
    if X.shape[0] < 500 or X.shape[1] < 12:
        raise ValueError("Dataset does not meet the assignment's size requirements.")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Unexpected missing/infinite values; investigate before training.")
    print("Dataset: Card Campaign Booking (Synthetic)")
    print("Source:", DATASET_SOURCE)
    print("Feature table shape:", X.shape)
    print("Target meaning: 0 = no booking; 1 = booked")
    print("Class counts:\n", y.value_counts().sort_index().to_string())
    print("Missing feature values:", int(X.isna().sum().sum()))
    print("First five rows, showing five of the 17 features:\n", X.iloc[:5, :5])
    state.update(X=X, y=y)
    print("LEARN: rows are samples; columns are measurements used for prediction.")

def step_3_split(state):
    heading(3, "Separate training rows from held-out test rows")
    X_train, X_test, y_train, y_test = train_test_split(
        state["X"], state["y"], test_size=TEST_FRACTION,
        random_state=SEED, stratify=state["y"],
    )
    state.update(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Training rows:", len(X_train), "| Test rows:", len(X_test))
    print("Training class counts:\n", y_train.value_counts().sort_index())
    print("Test class counts:\n", y_test.value_counts().sort_index())
    print("LEARN: do not fit a scaler or tune a model using the held-out test set.")

def step_4_train(state):
    heading(4, "Train five models and compare training CV AUC")
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=2, random_state=SEED)),
        ]),
        "kNN": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([("classifier", GaussianNB())]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            # probability=True enables predict_proba for AUC in the app.
            ("classifier", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
        ]),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    for name, model in models.items():
        print("Training:", name, flush=True)
        scores = cross_val_score(
            model, state["X_train"], state["y_train"], cv=folds,
            scoring="roc_auc", n_jobs=1, error_score="raise",
        )
        cv_rows.append({"ML Model Name": name, "CV AUC Mean": scores.mean(),
                        "CV AUC Std": scores.std()})
        # fit learns parameters from ALL training rows, never from test rows.
        model.fit(state["X_train"], state["y_train"])
    cv_table = pd.DataFrame(cv_rows)
    selected = cv_table.loc[cv_table["CV AUC Mean"].idxmax(), "ML Model Name"]
    state.update(models=models, cv_table=cv_table, selected_model=selected)
    print(cv_table.round(4).to_string(index=False))
    print("Selected using training CV AUC:", selected)

def measure_model(model, X, y):
    predictions = model.predict(X)
    # Find class 1's actual column instead of assuming the probability order.
    positive_column = list(model.classes_).index(1)
    positive_probability = model.predict_proba(X)[:, positive_column]
    auc = float(roc_auc_score(y, positive_probability)) if len(np.unique(y)) == 2 else np.nan
    metrics = {
        "Accuracy": float(accuracy_score(y, predictions)),
        "AUC": auc,
        "Precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predictions)),
    }
    return metrics, predictions, positive_probability


def step_5_evaluate(state):
    heading(5, "Evaluate all models on the same held-out test set")
    rows = []
    for name, model in state["models"].items():
        metrics, _, _ = measure_model(model, state["X_test"], state["y_test"])
        rows.append({"ML Model Name": name, **metrics})
    state["comparison"] = pd.DataFrame(rows)
    print("REQUIRED SIX-METRIC COMPARISON TABLE")
    print(state["comparison"].round(4).to_string(index=False))
    print("Selected model (training CV criterion):", state["selected_model"])
    print("LEARN: one split is an estimate, not a guarantee of future performance.")

def confusion_figure(y, predictions, title):
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def step_6_reports(state):
    heading(6, "Save charts and detailed reports")
    report_dir = state["output"] / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, model in state["models"].items():
        predictions = model.predict(state["X_test"])
        stem = Path(MODEL_FILES[name]).stem
        fig = confusion_figure(state["y_test"], predictions, name)
        fig.savefig(report_dir / f"{stem}_confusion_matrix.png", dpi=160)
        plt.close(fig)
        report = classification_report(
            state["y_test"], predictions, labels=[0, 1],
            target_names=CLASS_NAMES, zero_division=0,
        )
        (report_dir / f"{stem}_classification_report.txt").write_text(report, encoding="utf-8")
    print("Saved five PNG charts and five text reports in:", report_dir)


def markdown_table(frame):
    """Build a small Markdown table without needing the optional tabulate package."""
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_readme(state):
    """Generate an honest draft from measured values, with student work marked."""
    explanations = {
        "Logistic Regression": "Linear decision boundary with standardized inputs.",
        "Decision Tree": "Depth-limited nonlinear rules; inspect false negatives and overfitting.",
        "kNN": "Uses the five nearest standardized training examples.",
        "Naive Bayes": "Gaussian likelihoods; conditional independence is a simplifying assumption.",
        "Support Vector Machine": "RBF kernel captures nonlinear boundaries after standardization.",
    }
    observations = []
    for row in state["comparison"].to_dict("records"):
        name = row["ML Model Name"]
        note = (f"Test F1={row['F1']:.4f}, recall={row['Recall']:.4f}, "
                f"MCC={row['MCC']:.4f}. {explanations[name]} "
                "STUDENT: interpret this result and the confusion matrix in your own words.")
        observations.append({"ML Model Name": name, "Observation about model performance": note})
    observations.append({"ML Model Name": "Overall, Winner for your dataset?",
                         "Observation about model performance":
                         state["selected_model"] + " selected by mean training CV AUC, not test tuning. "
                         "STUDENT: discuss trade-offs and limitations."})
    return f"""# ML Assignment 2 - learning project (personalize before submission)

## a. Problem statement
Compare five binary classifiers for predicting whether a customer books a
card campaign offer within seven days of seeing it.

## b. Dataset description
Synthetic card campaign impressions, with one row per simulated customer.
Source: {DATASET_SOURCE}
The CSV contains {len(state['X'])} rows and {state['X'].shape[1]} numeric predictors.
Predictors describe card type, prior activity and offer terms known at impression time.
`booked_with_card_7d` is 1 when a booking was made with a card within seven days and 0 otherwise.
Binary precision, recall and F1 use booked as the positive class. Results describe simulated behavior.
Split: {len(state['X_train'])} training rows and {len(state['X_test'])} held-out test rows;
stratified 80/20 split, random seed {SEED}. All five models use the same rows.
Scalers are fitted inside pipelines and inside each CV training fold.
Model selection uses five-fold training CV AUC. Final test AUC uses class-1 probabilities.
SVM CV AUC uses its decision scores. No test-based hyperparameter tuning is performed.

## c. GitHub Repository Link
TODO: insert your actual clickable repository link after creating the repository.
Live Streamlit link: TODO: insert your actual deployed app URL.

## d. Models used
{markdown_table(state['comparison'])}

### Training-only model-selection evidence
{markdown_table(state['cv_table'])}

### Observations on each model
{markdown_table(pd.DataFrame(observations))}

These observations are a starting point. Inspect your own results, explain model
assumptions, and write an original interpretation. One held-out split is uncertain;
differences on this small dataset do not establish universal superiority.

## Run locally / in BITS Virtual Lab
Use Python {platform.python_version()} or a compatible Python version for these dependencies.
Install: `python -m pip install -r requirements.txt`
From the project root, retrain: `python model/train_models.py --output .`
From the project root, launch: `python -m streamlit run app.py`
If the lab does not expose the web port, use its supported proxy or deploy on Community Cloud.
The app loads saved pipelines; it does not retrain on uploaded data.
Use the supplied `test_data.csv`: 17 numeric features plus optional `booked_with_card_7d`.
Labels must be 0=no booking and 1=booked. Both labels are needed for AUC.
Without booked_with_card_7d, the app displays predictions and probabilities.

## Files
- `app.py`: commented training/app source; launched in app mode by its filename.
- `model/train_models.py`: full training source, with numbered teaching sections.
- `model/card_campaign_bookings.csv`: full dataset for retraining; use test_data.csv in the app.
- `model/*.joblib`: five fitted pipelines, including any required scalers.
- `test_data.csv`: held-out data only; not the full or training dataset.
- `requirements.txt`: dependency versions from the training environment.
- `comparison_metrics.csv`, `training_cv_auc.csv`, `metadata.json`: reproducible results.
- `reports/`: five confusion matrices and five detailed classification reports.

## Before submission (manual work)
1. Understand and adapt this learning reference; acknowledge assistance as required.
2. Personalize your app and write your own analysis; do not submit unchanged AI output.
3. Run on BITS Virtual Lab and take your own screenshot showing execution/results.
4. Create your own GitHub repository and commit the required project files.
5. Deploy on Streamlit Community Cloud with entry point `app.py`.
6. Replace the TODO links above and verify both links open correctly.
7. Prepare ONE submission PDF in this order: repository link, live app link,
   actual lab-execution screenshot, then the complete README content.
8. Submit the PDF through the course portal; this script does not submit it.

## References
- Dataset: {DATASET_SOURCE}
- Validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit: https://docs.streamlit.io/
"""

def step_7_export(state):
    heading(7, "Export the runnable project")
    root = state["output"]
    model_dir = root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in state["models"].items():
        joblib.dump(model, model_dir / MODEL_FILES[name])
    # Keep a full CSV with the exported training script for local retraining.
    full_data = state["X"].copy()
    full_data[TARGET] = state["y"]
    full_data.to_csv(model_dir / "card_campaign_bookings.csv", index=False)
    # The row indices align, so the target stays attached to its correct sample.
    test_data = state["X_test"].copy()
    test_data[TARGET] = state["y_test"]
    test_data.to_csv(root / "test_data.csv", index=False)
    state["comparison"].to_csv(root / "comparison_metrics.csv", index=False)
    state["cv_table"].to_csv(root / "training_cv_auc.csv", index=False)
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib"]
    versions = {name: package_version(name) for name in packages}
    info = {
        "dataset": "Card Campaign Booking (Synthetic)", "source": DATASET_SOURCE,
        "features": list(state["X"].columns), "target": TARGET,
        "class_mapping": {"0": "no booking", "1": "booked"}, "seed": SEED,
        "train_rows": len(state["X_train"]), "test_rows": len(state["X_test"]),
        "selected_by_training_cv_auc": state["selected_model"],
        "python_version": platform.python_version(), "package_versions": versions,
        "model_files": MODEL_FILES,
    }
    (root / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    requirements = [f"{name}=={version}" for name, version in versions.items() if version]
    streamlit_version = package_version("streamlit")
    requirements.append(f"streamlit=={streamlit_version}" if streamlit_version else "streamlit>=1.30,<2")
    (root / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(build_readme(state), encoding="utf-8")
    app = root / "app.py"
    if not app.exists():
        app.write_text(
            'from pathlib import Path\nfrom model import train_models\n\n'
            'if __name__ == "__main__":\n'
            '    train_models.load_ml_imports()\n'
            '    train_models.run_streamlit_app(Path(__file__).resolve().parent)\n',
            encoding="utf-8",
        )
    print("Project saved in:", root)
    print("Saved test rows:", len(test_data), "| Columns:", len(test_data.columns))
    print("Open comparison_metrics.csv for the current run.")

def step_8_next_actions(state):
    heading(8, "Next actions after reviewing your output")
    print("In a NEW TERMINAL, launch the generated app using:")
    print(f'python -m streamlit run "{state["output"] / "app.py"}"')
    print("Do not type this terminal command into a Python cell without !.")
    print("Keep the terminal running while using the app.")
    print("In the app: use bundled test data, select each model, then try CSV upload.")
    print("Take your own BITS Lab screenshot showing notebook/terminal results.")
    print("Review README, personalize, commit to your GitHub, and deploy app.py.")
    print("Final PDF needs: GitHub link, live app link, lab screenshot, README.")

def run_streamlit_app(root):
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install streamlit, then launch with: python -m streamlit run app.py") from exc
    st.set_page_config(page_title="Card Campaign Booking Lab", layout="wide")
    st.title("Card Campaign Booking Lab")
    st.caption("Five classifiers | one test set | transparent comparison")
    st.info("Synthetic campaign data. 1 = booked with a card within 7 days. 0 = no booking within 7 days.")

    @st.cache_resource
    def load_local_artifacts(root_path):
        folder = Path(root_path)
        info = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        models = {name: joblib.load(folder / "model" / filename)
                  for name, filename in info["model_files"].items()}
        return info, models

    try:
        info, models = load_local_artifacts(str(root))
    except Exception as exc:
        st.error(f"Could not load project files: {exc}. Run training/export step 7 first. "
                 "If versions differ, install requirements.txt or retrain in this environment.")
        st.stop()
    st.sidebar.header("Explore the held-out data")
    selected = st.sidebar.selectbox("Choose a classifier", list(models))
    use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)
    upload = st.sidebar.file_uploader("Upload test CSV (not training data)", type=["csv"])
    st.sidebar.caption("An uploaded file takes priority. Include booked_with_card_7d for evaluation metrics.")
    st.sidebar.write("Training-CV selected model:", info["selected_by_training_cv_auc"])
    sample_path = root / "test_data.csv"
    if sample_path.exists():
        st.sidebar.download_button("Download example test CSV", sample_path.read_bytes(),
                                   file_name="test_data.csv", mime="text/csv")
    try:
        if upload is not None:
            frame = pd.read_csv(upload)
            source = "Uploaded test CSV"
        elif use_bundled:
            frame = pd.read_csv(sample_path)
            source = "Bundled held-out test data"
        else:
            st.info("Upload test_data.csv or enable the bundled test data.")
            st.stop()
        X, y = validate_test_csv(frame, info["features"])
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()
    st.subheader(f"{source}: {len(X)} rows, {len(X.columns)} features")
    with st.expander("Inspect the data and metric definitions"):
        st.dataframe(frame.head(10))
        st.markdown("**Accuracy:** all correct predictions / all rows. "
                    "**Precision:** TP/(TP+FP). **Recall:** TP/(TP+FN). "
                    "**F1:** harmonic mean of precision and recall. "
                    "**AUC:** ranking across thresholds. **MCC:** label agreement, -1 to +1. "
                    "Positive class is booked (1). Matrix rows are actual; columns predicted.")
    model = models[selected]
    predictions = model.predict(X)
    positive_probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
    if y is not None:
        rows = []
        for name, candidate in models.items():
            metrics, _, _ = measure_model(candidate, X, y)
            rows.append({"ML Model Name": name, **metrics})
        comparison = pd.DataFrame(rows)
        st.subheader("All five models on the current test data")
        st.dataframe(comparison.round(4))
        st.download_button("Download current comparison", comparison.to_csv(index=False),
                           file_name="current_test_metrics.csv", mime="text/csv")
        selected_metrics = comparison.set_index("ML Model Name").loc[selected]
        st.subheader(selected)
        for column, (name, value) in zip(st.columns(6), selected_metrics.items()):
            column.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")
        if y.nunique() < 2:
            st.warning("AUC is undefined because these labels contain only one class.")
        fig = confusion_figure(y, predictions, selected)
        st.pyplot(fig)
        plt.close(fig)
        report = classification_report(y, predictions, labels=[0, 1],
                                       target_names=CLASS_NAMES, zero_division=0)
        st.subheader("Classification report")
        st.code(report, language="text")
    else:
        st.info("Include booked_with_card_7d to see evaluation metrics.")
    output = X.copy()
    if y is not None:
        output[TARGET] = y
    output["prediction"] = predictions
    output["predicted_label"] = np.where(predictions == 1, "booked", "no booking")
    output["probability_booked"] = positive_probability
    st.subheader("Predictions from " + selected)
    st.dataframe(output.head(20))
    st.download_button("Download all predictions", output.to_csv(index=False),
                       file_name="predictions.csv", mime="text/csv")




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=range(1, 9), default=8,
                        help="Run steps 1 through this number (default: 8).")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter between steps; terminal only.")
    parser.add_argument("--output", type=Path, help="Output folder; use a fresh folder for a new experiment.")
    parser.add_argument("--csv", type=Path,
                        default=Path(__file__).resolve().parent / "card_campaign_bookings.csv",
                        help="Full dataset CSV; labels must be 0=no booking, 1=booked.")
    parser.add_argument("--app", action="store_true", help="App mode for the original learning filename.")
    args = parser.parse_args(argv)
    is_exported_app = Path(__file__).name == "app.py"
    root = args.output or (Path(__file__).resolve().parent if is_exported_app
                           else Path(__file__).resolve().parent.parent)
    state = {"output": root.resolve(), "csv": args.csv.expanduser().resolve()}
    if args.app or is_exported_app:
        load_ml_imports()
        run_streamlit_app(state["output"])
        return
    steps = [step_1_environment, step_2_dataset, step_3_split, step_4_train,
             step_5_evaluate, step_6_reports, step_7_export, step_8_next_actions]
    for number, function in enumerate(steps, start=1):
        if number > args.step:
            break
        if number == 2:
            load_ml_imports()
        function(state)
        if args.pause and number < args.step:
            input("Press Enter for the next step (Ctrl+C stops)... ")


if __name__ == "__main__" and "__file__" in globals():
    try:
        main()
    except (RuntimeError, ValueError, ModuleNotFoundError, OSError) as error:
        print("\nSTOP:", error, file=sys.stderr)
        print("Fix the issue, then rerun the same step. Share the full error if needed.", file=sys.stderr)
        raise SystemExit(1)
