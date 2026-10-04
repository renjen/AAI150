# Final Report Outline

Final file: `Final-Project-Report-Team-X.pdf` (replace X with our team number)

Owners follow the work split in the README. Intro and Conclusion are written together at the end.

## Checklist

- [ ] Title page
- [ ] 1. Introduction (everyone)
- [x] 2. Data Cleaning/Preparation (Person A), draft in `02_data_cleaning.md`
- [x] 3. Exploratory Data Analysis (Person B), draft in `03_eda.md`
- [ ] 4. Model Selection (Person C)
- [ ] 5. Model Analysis (Person C)
- [ ] 6. Conclusion and Recommendations (everyone)
- [ ] References
- [ ] Appendix: notebook output
- [ ] Final read-through so it sounds like one report
- [ ] Turnitin / Draft Coach check
- [ ] Export to PDF and submit

---

## Title Page

- Project title (e.g. "Predicting Wine Quality from Chemical Properties")
- Team number, all team member names
- Course (AAI 150), instructor, date
- GitHub repo link: https://github.com/renjen/AAI150

## 1. Introduction (everyone)

Write this last, after the results are in.

- The problem: wine quality is usually judged by expert tasters, which is slow and subjective. Can lab measurements predict it?
- Why it matters to a business (winemakers, distributors, retailers): quality control, pricing, deciding which batches to promote
- The dataset in a few sentences (UCI Wine Quality, 6,497 red and white Vinho Verde wines, 11 chemical measurements, expert quality score)
- Our goal: classify wines as good (quality ≥ 7) or not good
- Quick preview of the main result (best model and how well it did)
- One sentence on how the rest of the report is organized

## 2. Data Cleaning/Preparation (Person A)

Draft done: see `02_data_cleaning.md`.

- Data sources and loading
- Combining red and white, adding `type`
- Data types and missing values
- Duplicates (1,177 removed)
- Outliers (checked with IQR, kept)
- Target variable `is_good` and class imbalance
- Train/test split
- Summary table

## 3. Exploratory Data Analysis (Person B)

Use `data/wine_clean.csv`.

- Summary statistics table for all features (mean, std, min, max)
- Distribution of the target: bar chart of good vs. not good, and of the original quality scores
- Distributions of each feature (histograms), noting which ones are skewed (residual sugar, chlorides, sulfur dioxide)
- Red vs. white comparison: which features differ the most
- Correlation heatmap of all features
  - Which features are strongly correlated with each other (e.g. free and total sulfur dioxide, density and alcohol)
  - Which features relate most to `is_good`
- Box plots of the top features split by good vs. not good (alcohol is expected to stand out)
- Key takeaways that lead into modeling:
  - Which features look most useful
  - Any multicollinearity to worry about
  - Class imbalance reminder

## 4. Model Selection (Person C)

Use `data/train.csv` and `data/test.csv`. Drop `quality` and `type` from the inputs.

- Preprocessing: feature scaling (fit on training data only) and how class imbalance is handled (e.g. `class_weight="balanced"`)
- Candidate models and why each was chosen, for example:
  - Logistic regression (simple baseline, easy to explain)
  - Random forest
  - Gradient boosting (or another model)
- How models were compared: cross-validation on the training set
- Metrics used and why (precision, recall, F1, ROC-AUC, not just accuracy, because only 19% of wines are good)
- Hyperparameter tuning, if any (e.g. grid search)
- Comparison table of all models
- Which model was picked and the statistical reasoning behind it

## 5. Model Analysis (Person C)

- Final model performance on the test set
- Confusion matrix and what the errors mean (good wines missed vs. average wines labeled good)
- ROC curve and AUC
- Precision-recall curve (useful because of the imbalance)
- Feature importance: which chemical properties matter most
- Is the model valid? Check for overfitting (train vs. test scores) and whether results hold up across cross-validation folds
- Limitations:
  - Only Vinho Verde wines, so results may not generalize
  - Quality scores are subjective
  - No price, grape type, or brand information in the data

## 6. Conclusion and Recommendations (everyone)

Write this for a business reader.

- Short recap of what we did and the main result
- The chemical properties that most drive quality, in plain language
- Recommendations, for example:
  - Use the model as a first screening step before expert tasting
  - Which properties winemakers could monitor or adjust
- Future work: more data, other wine regions, predicting the exact score, trying other models

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553.
- UCI Machine Learning Repository: Wine Quality. https://archive.ics.uci.edu/dataset/186/wine+quality
- Any libraries or other sources we cite (pandas, scikit-learn, etc.)

## Appendix: Technical Notebook

- Required by the assignment: the output of our code from the Jupyter notebook(s)
- Put all notebooks in order (cleaning, EDA, modeling) so it reads as one story
- Export to PDF (in VS Code or Jupyter: File > Export / Save As PDF) and attach it to the end of the report

---

## Video Presentation (separate deliverable, for planning)

File: `Final-Project-Presentation-Team-X.mp4`, 8 to 10 minutes, no longer than 10.

- Aimed at non-technical business leaders: no code, few formulas
- Everyone presents an equal share (about 3 minutes each)
- Suggested flow:
  1. The problem and why it matters
  2. The data and how we cleaned it (Person A)
  3. What the data showed (Person B)
  4. The model and how well it works (Person C)
  5. Recommendations
  6. Team contributions slide (required): each member's name and what they did
