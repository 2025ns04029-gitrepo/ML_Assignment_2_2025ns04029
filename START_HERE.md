# Card campaign booking

This project uses 2,000 synthetic card campaign impressions and 17 features. The assignment asks for a Kaggle or UCI dataset. Confirm that your instructor accepts this dataset before submitting.


Booking_Campaign_Assignment/
  START_HERE.md
  project/                              GitHub repository root
    .gitignore
    Booking_Campaign_with_Cards.ipynb                Run in BITS Virtual Lab
    README.md                            Dataset, results and your observations
    app.py                               Streamlit entry point
    requirements.txt                     App and model dependencies
    metadata.json                        Feature order, labels and model names
    test_data.csv                        400 held-out rows for the app
    comparison_metrics.csv               Five models and six test metrics
    training_cv_auc.csv                  Training-fold AUC summary
    model/
      train_models.py                    Shared training and app functions
      generate_dataset.py                Reproduces the synthetic dataset
      card_campaign_bookings.csv                  Full synthetic dataset for training
      logistic_regression.joblib
      decision_tree.joblib
      knn.joblib
      naive_bayes.joblib
      svm.joblib
    reports/
      logistic_regression_confusion_matrix.png
      logistic_regression_classification_report.txt
      decision_tree_confusion_matrix.png
      decision_tree_classification_report.txt
      knn_confusion_matrix.png
      knn_classification_report.txt
      naive_bayes_confusion_matrix.png
      naive_bayes_classification_report.txt
      svm_confusion_matrix.png
      svm_classification_report.txt
  submission/                           Keep local
    Assignment_2_Draft.pdf               Preview of the required PDF order
    build_submission_pdf.py             Builds your final PDF
```

## 1. Run in BITS Virtual Lab

Upload the `project/` folder and preserve its subfolders. Open `Booking_Campaign_with_Cards.ipynb` inside it. Use the Python 3 kernel and run the cells from top to bottom. If packages are missing, run the commented install line in Step 1. Use Python 3.11 or later with the provided dependency versions.

Step 5 displays the live comparison. Capture one screenshot showing the BITS Virtual Lab interface and your executed comparison output. Save it as `submission/bits_lab_screenshot.png` on your computer. Step 7 exports the app files and preserves your existing README.

## 2. Verify and publish the app

From a terminal in `project/`, run the following commands if testing the app locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Check that the app displays all five models and six metrics. Upload `test_data.csv`, select a model, and inspect its confusion matrix and report. The file in `model/card_campaign_bookings.csv` is the full dataset for training.

Open the GitHub repository you created and commit the contents of `project/` at its root. In Streamlit Community Cloud, choose that repository, its branch, and `app.py`. Copy the live app URL once it opens. Replace `{GITHUB_URL}` and `{STREAMLIT_URL}` in `project/README.md`, write your own observations based on your lab run, and commit the README. If your lab run produces different values, update the table in the README from your generated `comparison_metrics.csv`.

If you prefer Git in Terminal, start from the extracted `Booking_Campaign_Assignment/` folder. Replace `YOUR-NAME` and `YOUR-REPO` with the repository you created:

```bash
git clone https://github.com/YOUR-NAME/YOUR-REPO.git repo
cp -R -n project/. repo/
cd repo
git status --short
git add .
git commit -m "Add card campaign booking classifier project"
git push
```

The copy command skips files with the same name. If your repo already contains a README or an app from another project, review and merge those files before committing. On GitHub, `app.py`, `requirements.txt`, and the `model/` folder should appear at the same level.

## 3. Build the single submission PDF

From the extracted `Booking_Campaign_Assignment/` folder on your Mac, run these commands after adding your actual links, screenshot, name and BITS ID:

```bash
python3 -m pip install reportlab pillow
python3 submission/build_submission_pdf.py \
  --github-url "https://github.com/YOUR-NAME/YOUR-REPO" \
  --app-url "https://YOUR-APP.streamlit.app" \
  --screenshot submission/bits_lab_screenshot.png \
  --name "YOUR NAME" \
  --bits-id "YOUR BITS ID"
```

Open `submission/Assignment_2_Final.pdf`. Verify both links and check that the screenshot is readable. The PDF puts the GitHub link first, the live app link second, the screenshot third, and the README content fourth. Upload the PDF through the course portal.
