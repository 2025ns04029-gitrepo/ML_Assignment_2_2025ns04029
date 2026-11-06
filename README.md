# ML Assignment 2 | Card Campaign Booking Prediction

## a. Problem statement

Predict whether a customer will book a campaign offer using a card within seven days of seeing the offer. The comparison illustrates how the choice of evaluation metric changes which model looks strongest.

## b. Dataset description

The dataset is synthetic and reproducible with seed 42. It contains 2,000 campaign impressions and 17 numeric predictors. Each row represents one offer shown to one simulated customer. The predictors describe card type, prior card and offer activity, and offer terms known at impression time. The target is `booked_with_card_7d`, where 1 means a card booking within seven days and 0 means no card booking in that window. `model/generate_dataset.py` documents the simulation.

The assignment asks for a public dataset from Kaggle or UCI. Confirm that your instructor accepts this synthetic dataset before submitting this project.

All models use the same 1,600 training records and 400 held-out test records. The test set has 122 card bookings and 278 rows without a booking. The split is stratified with seed 42. Scaling for Logistic Regression, kNN and SVM is fitted inside each training pipeline. ROC AUC uses booking probabilities, while precision, recall and F1 use booking as the positive class.

## c. GitHub repository link

GitHub repository: {GITHUB_URL}

Live Streamlit app: {STREAMLIT_URL}

## d. Models and comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7225 | 0.7323 | 0.5821 | 0.3197 | 0.4127 | 0.2700 |
| Decision Tree | 0.6875 | 0.6269 | 0.4776 | 0.2623 | 0.3386 | 0.1682 |
| kNN | 0.6725 | 0.6057 | 0.4416 | 0.2787 | 0.3417 | 0.1448 |
| Gaussian Naive Bayes | 0.6950 | 0.7072 | 0.5000 | 0.4098 | 0.4505 | 0.2445 |
| Support Vector Machine | 0.7100 | 0.6879 | 0.5833 | 0.1721 | 0.2658 | 0.1901 |

### Model observations

| Model | Observation to verify in your lab run |
| --- | --- |
| Logistic Regression | Highest held-out AUC and accuracy. Recall was 0.3197 for card bookings. |
| Decision Tree | Recall was 0.2623. The tree found fewer bookings than Logistic Regression. |
| kNN | Had the lowest AUC. Its recall was slightly higher than the Decision Tree. |
| Gaussian Naive Bayes | Highest recall and F1. It found about 41% of actual bookings in this split. |
| Support Vector Machine | Highest precision and lowest recall. It predicted relatively few bookings. |

Overall choice for this run: Logistic Regression had the highest mean AUC across five training folds (0.7292). Gaussian Naive Bayes had the best held-out F1 (0.4505). Verify these observations with your own BITS Virtual Lab output and write your interpretation of the trade-off.

## Run the project

Open `Booking_Campaign_with_Cards.ipynb` from this folder. Run the cells from top to bottom. Use `python model/train_models.py --output .` for a terminal run.

Install with `python -m pip install -r requirements.txt` and launch with `python -m streamlit run app.py`. The app can compare all five models on the bundled `test_data.csv` or a labeled CSV with the same schema. Select a model to see its confusion matrix and classification report.

`model/` contains the full dataset, generator, training code and five saved pipelines. `reports/` contains per-model results. `comparison_metrics.csv` holds the six held-out metrics and `training_cv_auc.csv` holds training cross-validation results.
