# Modeling Wine Quality Using Physicochemical Attributes

**Final Project Report, Team X**

Renee Dhanaraj  
Person B Name  
Anthony Candelas

AAI 150  
M.S. Applied Artificial Intelligence, University of San Diego  
Instructor: Instructor Name  
October 9, 2026

GitHub repository: https://github.com/renjen/AAI150

---

# Introduction

The quality of a wine is usually judged by expert tasting panels. Tasting is slow and expensive, and the scores are subjective: two experts can rate the same wine differently, and a panel can only taste so many wines in a day. Premium wines sell at much higher prices than ordinary table wines, so deciding which batches are premium matters a great deal to a winery. Laboratory tests, in contrast, are fast, cheap, and objective, and wineries already run them for quality control and certification. If lab measurements can predict how experts will rate a wine, a winery could use them to screen batches early, decide which wines to send to expert tasting, and focus its pricing and marketing on its best products.

In this project, we test whether a wine's chemical properties can predict whether experts will rate it highly. We use the Wine Quality dataset from the UCI Machine Learning Repository (Cortez et al., 2009). It contains 6,497 red and white Vinho Verde wines from northern Portugal. Each wine has 11 physicochemical measurements, such as alcohol content, acidity, residual sugar, and sulfur dioxide, and a quality score from 0 to 10, which is the median rating of at least three experts.

We framed the problem as **binary classification**: a wine is **good** if its quality score is 7 or higher and **not good** otherwise. This matches a common business question, "is this batch one of our best?", and is easier to act on than predicting an exact score. Only 19% of the wines are good, so a central challenge is building a model that finds this small group without labeling too many average wines as good.

The analysis addresses three questions:

1. Which model gives the best balance of recall, precision, and reliability on new wines, given that good wines are rare?
2. Which chemical properties are most strongly linked to a high quality rating?
3. How could a winery use the model's predictions in practice, for example to screen batches and decide which wines to send to expert tasting?

After removing 1,177 duplicate records, we compared logistic regression, random forest (Breiman, 2001), and gradient boosting models using 10-fold cross-validation. **Our final model, a random forest, achieved a ROC-AUC of 0.874 and an F1 score of 0.617 on a held-out test set of 1,064 wines, finding 72% of the good wines.** It was significantly better than logistic regression and performed about as well as gradient boosting. Alcohol content was by far the most important predictor, followed by chlorides and density.

The rest of the report is organized as follows. The Data Cleaning and Preparation section describes how we combined and cleaned the data. The Exploratory Data Analysis section describes the patterns we found. The Model Selection section explains how we compared and chose models, and the Model Analysis section evaluates the final model and its limitations. We close with conclusions and recommendations for a winery. The code and its output are in the Appendix.

---

# Data Cleaning and Preparation

## Data Sources

The data comes from the Wine Quality dataset in the UCI Machine Learning Repository (Cortez et al., 2009). It has two files, one for red wines (1,599 samples) and one for white wines (4,898 samples), all from the Vinho Verde region of Portugal. Each sample has 11 physicochemical measurements taken in the lab: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol. Each wine also has a quality score from 0 to 10, which is the median of at least three ratings from wine experts. In practice, the scores only range from 3 to 9.

Both files are semicolon-separated, so we set the separator when loading them with pandas.

## Combining Red and White Wines

We combined the red and white files into a single dataset of 6,497 rows instead of analyzing them separately. This gives the models more data to learn from, and it lets us compare the two wine types directly. To keep track of where each row came from, we added a `type` column ("red" or "white") before combining. We also renamed the columns to snake_case (for example, `fixed acidity` became `fixed_acidity`) so they are easier to use in code.

## Data Types and Missing Values

All 11 measurements loaded as floating-point numbers, `quality` loaded as an integer, and `type` as text. These are the types we expected, so no conversions were needed.

We found no missing values in any column, so no rows had to be removed or filled in for this reason.

## Duplicate Records

We found 1,177 rows that were exact copies of another row, meaning every measurement, the quality score, and the wine type all matched. This was 240 red wines and 937 white wines, about 18% of the combined data.

We decided to remove the duplicates and keep only the first copy of each. With 11 measurements recorded to several decimal places, it is unlikely that this many different wines would match on every value, so most of these rows are probably repeated entries of the same sample. Keeping them would also cause a problem later: when the data is split, identical rows could end up in both the training and test sets. The model would then be tested partly on data it had already seen, and its performance would look better than it really is. After removing duplicates, 5,320 wines remained (1,359 red and 3,961 white).

## Outliers

We checked each measurement for outliers using the interquartile range (IQR) rule, where a value is an outlier if it falls more than 1.5 times the IQR below the first quartile or above the third quartile. Table 1 shows the results.

**Table 1.** Outliers per feature using the 1.5 × IQR rule (after removing duplicates, n = 5,320)

| Feature | Outliers | Feature | Outliers |
|---|---|---|---|
| Fixed acidity | 304 | pH | 49 |
| Volatile acidity | 279 | Free sulfur dioxide | 44 |
| Chlorides | 237 | Total sulfur dioxide | 10 |
| Sulphates | 163 | Density | 3 |
| Citric acid | 143 | Alcohol | 1 |
| Residual sugar | 141 | | |

In total, 1,094 wines had at least one outlying value. We also compared each feature across red and white wines using box plots (see Appendix).

We decided to keep all of the outliers, for three reasons:

1. **Many of them come from combining two different types of wine.** Red and white wines have different chemistry. For example, the median volatile acidity is 0.52 g/L for red wines and 0.26 g/L for white wines, and the median total sulfur dioxide is 38 mg/L for red and 133 mg/L for white. A value that looks extreme for the combined data can be normal for its own wine type.
2. **The extreme values are realistic.** The highest residual sugar (65.8 g/L) matches a sweet wine, and the highest sulfur dioxide values (289 mg/L free, 440 mg/L total) are high but possible. None of the values are impossible, such as negative concentrations, so they are most likely real wines and not data entry errors.
3. **Removing them would lose information.** About one in five wines has at least one outlier, so dropping them would shrink the dataset a lot and could remove exactly the unusual wines that are useful to learn from. If outliers become a problem for a specific model, feature scaling or tree-based models can handle them.

## Target Variable

Since our goal is classification, we turned the quality score into a binary target called `is_good`. A wine is labeled good (`is_good = 1`) if its quality is 7 or higher, and not good (`is_good = 0`) otherwise. After cleaning, 1,009 wines (19.0%) are labeled good and 4,311 (81.0%) are not good. Good wines are more common among white wines (20.8%) than red wines (13.5%).

The classes are imbalanced. A model that always predicts "not good" would already be about 81% accurate, so accuracy alone is not a good way to judge our models. We use precision, recall, F1 score, and ROC-AUC in the model evaluation instead.

We also added a numeric column `is_red` (1 for red, 0 for white) so the models can use wine type as an input. The original `quality` column is not used as a model input, because `is_good` is calculated directly from it.

## Train/Test Split

We split the cleaned data into a training set (80%, 4,256 wines) and a test set (20%, 1,064 wines). The split was stratified on `is_good`, so both sets have the same share of good wines (19.0%). We used a fixed random seed (42) so every team member gets the same split and the results can be reproduced.

We did not scale the features at this stage. A scaler has to be fit only on the training data, otherwise information from the test set leaks into training. Scaling is done as part of the modeling step instead.

## Summary

**Table 2.** Summary of cleaning steps

| Step | Result |
|---|---|
| Load and combine | 6,497 wines (1,599 red, 4,898 white), `type` column added |
| Data types | All as expected, no changes |
| Missing values | None found |
| Duplicates | 1,177 removed, 5,320 wines remain |
| Outliers | Found with the IQR rule and kept |
| Target | `is_good` = 1 if quality ≥ 7 (19.0% of wines) |
| Train/test split | 80/20, stratified on `is_good`, 4,256 train and 1,064 test |

## Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553.

---

# Exploratory Data Analysis

We explored the full cleaned dataset (`wine_clean.csv`, 5,320 wines) to understand how each feature is distributed, how red and white wines differ, and which features are related to wine quality. The goal was to find out which features are likely to be useful for modeling and which problems the models would need to handle.

## Summary Statistics

Table 3 shows summary statistics for the 11 chemical features. The features are measured on very different scales. Density varies by less than 0.06 g/cm³ across all wines, while total sulfur dioxide ranges from 6 to 440 mg/L. For several features the mean is noticeably larger than the median, especially residual sugar (5.05 vs. 2.70 g/L), which is an early sign of right skew.

**Table 3.** Summary statistics of the chemical features (n = 5,320)

| Feature | Mean | Std. dev. | Min | Median | Max |
|---|---|---|---|---|---|
| Fixed acidity (g/L) | 7.215 | 1.320 | 3.800 | 7.000 | 15.900 |
| Volatile acidity (g/L) | 0.344 | 0.168 | 0.080 | 0.300 | 1.580 |
| Citric acid (g/L) | 0.318 | 0.147 | 0.000 | 0.310 | 1.660 |
| Residual sugar (g/L) | 5.048 | 4.500 | 0.600 | 2.700 | 65.800 |
| Chlorides (g/L) | 0.057 | 0.037 | 0.009 | 0.047 | 0.611 |
| Free sulfur dioxide (mg/L) | 30.037 | 17.805 | 1.000 | 28.000 | 289.000 |
| Total sulfur dioxide (mg/L) | 114.109 | 56.774 | 6.000 | 116.000 | 440.000 |
| Density (g/cm³) | 0.995 | 0.003 | 0.987 | 0.995 | 1.039 |
| pH | 3.225 | 0.160 | 2.720 | 3.210 | 4.010 |
| Sulphates (g/L) | 0.533 | 0.150 | 0.220 | 0.510 | 2.000 |
| Alcohol (% vol.) | 10.549 | 1.186 | 8.000 | 10.400 | 14.900 |

## Distribution of the Target

Figure 1 shows the original quality scores. Most wines received a 5 (1,752 wines) or a 6 (2,323 wines), and very few received the extreme scores of 3 (30 wines) or 9 (5 wines). Using quality ≥ 7 as the cutoff, 1,009 wines (19.0%) are good and 4,311 (81.0%) are not good. Good wines are more common among white wines (20.8%) than red wines (13.5%).

![Figure 1](figures/f01_quality_scores.png)

**Figure 1.** Distribution of quality scores. Scores of 7 and above are labeled good.

The imbalance means a model that always predicts "not good" would be 81% accurate while being useless. This confirms that accuracy should not be the main metric for comparing models.

## Feature Distributions

Figure 2 shows a histogram of each feature, and the skewness values confirm what the plots show. Most features are right-skewed, with a long tail of high values. Chlorides is by far the most skewed (skewness 5.34), followed by sulphates (1.81), residual sugar (1.71), fixed acidity (1.65), volatile acidity (1.50), and free sulfur dioxide (1.36). pH, citric acid, alcohol, and density are only mildly skewed.

Total sulfur dioxide has a skewness close to zero (0.06), but its histogram has two peaks. This is not a single symmetric distribution. It is two distributions on top of each other, one for red wines and one for white wines, as the next section shows.

![Figure 2](figures/f02_feature_histograms.png)

**Figure 2.** Histograms of the 11 chemical features.

Skewed features with very different scales can cause problems for models that rely on distances or linear combinations of the inputs, such as logistic regression. For these models we standardize the features. Tree-based models split on one feature at a time and are not affected by skew or scale.

## Red vs. White Wines

Red and white wines have very different chemistry (Figure 3). The biggest differences are in sulfur dioxide and sugar. White wines have a median total sulfur dioxide of 133 mg/L compared with 38 mg/L for red wines, more than twice as much free sulfur dioxide (33 vs. 14 mg/L), and more than twice as much residual sugar (4.7 vs. 2.2 g/L). Red wines have about twice the volatile acidity (0.52 vs. 0.26 g/L) and chlorides (0.079 vs. 0.042 g/L), and higher sulphates and fixed acidity. Alcohol, density, and pH are similar for both types.

![Figure 3](figures/f03_red_vs_white.png)

**Figure 3.** Each feature by wine type.

These differences explain many of the outliers found during data cleaning. A value that is normal for one type of wine can look extreme in the combined data. Their quality scores, on the other hand, are similar: the average quality is 5.85 for white wines and 5.62 for red wines. Because wine type affects so many features, we kept `is_red` as a model input so the models can take these differences into account.

## Correlations

Figure 4 shows the correlation matrix of all numeric variables. Two pairs of features are strongly correlated with each other:

- **Free and total sulfur dioxide** (r = 0.72). Free sulfur dioxide is part of the total, so this is expected.
- **Density and alcohol** (r = −0.67). Alcohol is less dense than water, so wines with more alcohol are lighter. Density is also related to residual sugar (r = 0.52), since dissolved sugar makes wine denser.

![Figure 4](figures/f04_correlation_heatmap.png)

**Figure 4.** Correlation matrix of all numeric variables.

This multicollinearity does not hurt a model's predictions much, but it can make the coefficients of a linear model unstable and hard to interpret, because the model cannot tell which of two related features is responsible for an effect. We keep this in mind when interpreting the logistic regression model in the Model Analysis section.

Figure 5 shows how each feature is correlated with our target, `is_good`. (The `quality` column is left out because `is_good` is calculated from it.) **Alcohol has by far the strongest relationship with being a good wine (r = 0.42).** Density (r = −0.30), chlorides (r = −0.16), and volatile acidity (r = −0.14) have the strongest negative relationships. The remaining features all have correlations close to zero (|r| < 0.09). On their own they say little about whether a wine is good, although they may still help in combination with other features.

![Figure 5](figures/f05_target_correlation.png)

**Figure 5.** Correlation of each feature with `is_good`.

## Features by Quality Class

Table 4 compares the average feature values of good and not good wines. Good wines have more alcohol (11.57% vs. 10.31%), lower volatile acidity (0.294 vs. 0.356 g/L), lower chlorides (0.044 vs. 0.060 g/L), lower density, and less residual sugar and total sulfur dioxide. Sulphates, pH, citric acid, and free sulfur dioxide are almost the same in both groups.

**Table 4.** Average feature values by quality class

| Feature | Not good | Good |
|---|---|---|
| Fixed acidity (g/L) | 7.247 | 7.079 |
| Volatile acidity (g/L) | 0.356 | 0.294 |
| Citric acid (g/L) | 0.314 | 0.337 |
| Residual sugar (g/L) | 5.233 | 4.259 |
| Chlorides (g/L) | 0.060 | 0.044 |
| Free sulfur dioxide (mg/L) | 29.939 | 30.453 |
| Total sulfur dioxide (mg/L) | 115.958 | 106.211 |
| Density (g/cm³) | 0.995 | 0.993 |
| pH | 3.221 | 3.241 |
| Sulphates (g/L) | 0.531 | 0.545 |
| Alcohol (% vol.) | 10.310 | 11.573 |

Figure 6 shows the clearest difference, alcohol. The middle half of good wines falls between 10.8% and 12.4% alcohol (median 11.6%), while the middle half of not good wines falls between 9.5% and 11.0% (median 10.1%). There is still overlap between the groups, so alcohol alone cannot separate good wines from the rest. Box plots of volatile acidity and sulphates, and scatterplots of alcohol and volatile acidity against the quality score, are in the Appendix.

![Figure 6](figures/f06_alcohol_boxplot.png)

**Figure 6.** Alcohol content of good and not good wines.

All of these relationships are associations. They do not show that changing a feature would change a wine's quality.

## Key Takeaways for Modeling

1. **Class imbalance.** Only 19% of wines are good. Models should be trained with class weights and judged on precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy.
2. **Most useful features.** Alcohol is the strongest single predictor, followed by density, chlorides, and volatile acidity. No single feature separates the classes well, so a model that combines features is needed.
3. **Skew and scale.** Most features are right-skewed and on very different scales, so logistic regression needs standardized inputs. Tree-based models do not.
4. **Multicollinearity.** Free and total sulfur dioxide, and density, alcohol, and residual sugar, are correlated. This matters most when interpreting linear model coefficients.
5. **Wine type.** Red and white wines have very different chemistry, so `is_red` is kept as an input.

---

# Model Selection

In this phase, we built and evaluated machine learning classifiers to predict whether a wine is high quality based on its 11 physicochemical measurements and wine type. Following the definition from data cleaning, a wine is good (`is_good = 1`) if its sensory evaluation score is 7 or higher, and ordinary (`is_good = 0`) if its score is 6 or lower.

The feature set includes 12 variables:
- 11 continuous chemical features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.
- 1 binary indicator: `is_red` (1 for red wine, 0 for white wine).

To prevent target leakage, the original continuous `quality` score and the text `type` column were dropped before modeling.

---

## Preprocessing and Prevention of Data Leakage

Data leakage happens when information from outside a training fold is used to transform features or select parameters. To prevent leakage:

1. **Strict Partitioning:** The 80/20 stratified split from data cleaning was strictly preserved. All model exploration, scaling, parameter estimation, and cross-validation were conducted exclusively on `train.csv` (N = 4,256). The test partition (`test.csv`, N = 1,064) remained completely untouched until the final evaluation.
2. **Encapsulated Preprocessing Pipelines:** For distance-sensitive algorithms requiring feature normalization (specifically Logistic Regression), standard z-score normalization (`StandardScaler`) was encapsulated within a scikit-learn `Pipeline`. During 10-fold cross-validation, the scaler was fit solely on the 9 training folds in each split and then applied to transform the held-out validation fold. Tree-based models (Random Forest and Gradient Boosting) do not require feature scaling, so they operate directly on the raw chemical values.

---

## Handling Class Imbalance

As documented in data cleaning, the target distribution is notably imbalanced:
- Ordinary Wines (`is_good = 0`): 81.04% of training observations (3,449 samples).
- High-Quality Wines (`is_good = 1`): 18.96% of training observations (807 samples).

A baseline classifier that always predicts the majority class would achieve 81.04% accuracy while failing completely to detect a single high-quality wine. To address this without distorting the data with synthetic oversampling:

1. **Stratified Partitioning:** A 10-fold Stratified K-Fold cross-validation scheme (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`) was used for all model comparisons. This guarantees that each validation fold contains the exact 18.96% positive class proportion.
2. **Cost-Sensitive Weighting:** All candidate models incorporated cost-sensitive loss weighting (`class_weight='balanced'`). The loss associated with misclassifying a minority-class observation was scaled inversely to class frequency according to:

$$w_j = \frac{N}{2 \times N_j}$$

where $N$ is the total number of training samples and $N_j$ is the count of samples in class $j$. This assigns a weight of approximately 2.64 to high-quality wines and 0.62 to ordinary wines, penalizing false negatives and compelling the algorithms to balance precision and recall.

---

## Evaluation Metrics

Because the classes are imbalanced, classification accuracy is uninformative. We evaluate each model using five metrics:

1. **Precision:** $\frac{TP}{TP + FP}$. The share of predicted high-quality wines that are truly high quality. High precision protects brand reputation by preventing ordinary wines from being bottled as premium reserves.
2. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$. The share of actual high-quality wines successfully identified by the model. High recall minimizes lost revenue by preventing premium wines from being sold as table wine.
3. **F1-Score:** The harmonic mean of precision and recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$). F1-score serves as our primary selection metric because it balances precision and recall without sacrificing either one.
4. **ROC-AUC:** Area Under the Receiver Operating Characteristic curve. Measures overall ranking and discrimination across all classification thresholds, independent of class prevalence.
5. **PR-AUC (Average Precision):** Area Under the Precision-Recall curve. Focuses specifically on the minority class trade-off across all thresholds, evaluated against the 0.1896 random baseline.

---

## Candidate Models

We benchmarked three model families to find the best balance of predictive power, resistance to overfitting, and interpretability:

1. **Logistic Regression (Parametric Baseline):**
   - *Rationale:* A linear reference baseline. It models log-odds as a linear combination of normalized inputs with $L_2$ ridge regularization ($C = 1.0$) and balanced class weights.
   - *Limitations:* Assumes linear decision boundaries and cannot capture non-linear chemical interactions (such as synergies between alcohol, sugar, and volatile acidity) without manual interaction terms.
2. **Random Forest Classifier (Ensemble Bagging: Primary Model):**
   - *Rationale:* An ensemble of 300 decorrelated decision trees built on bootstrap samples with random feature subsampling. Random forest handles skewed chemical distributions, resists outliers, captures non-linear interactions, and mitigates single-tree variance through bagging.
   - *Regularization:* Constrained to a maximum tree depth of 15, minimum split size of 4, and minimum leaf size of 2 to prevent memorizing noise in expert ratings.
3. **Histogram-Based Gradient Boosting (Ensemble Boosting: Comparative Model):**
   - *Rationale:* An iterative sequential boosting model that bins continuous features into 256 integer histograms, optimizing logistic loss through gradient descent.
   - *Configuration:* Evaluated with 200 boosting iterations, a learning rate of 0.08, maximum tree depth of 6, and balanced sample weighting.

---

## Cross-Validation Benchmark

Table 5 summarizes the cross-validation performance of all three candidate architectures across the 10 stratified validation folds. All metrics report the fold mean plus or minus one standard deviation.

**Table 5.** 10-Fold Stratified Cross-Validation Benchmark on Training Dataset (N = 4,256)

| Model Architecture | F1-Score | ROC-AUC | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.5316 +/- 0.0263 | 0.8306 +/- 0.0195 | 0.5282 +/- 0.0461 | **0.7869 +/- 0.0250** | 0.4021 +/- 0.0282 | 0.7361 +/- 0.0251 |
| **Random Forest (Final Model)** | **0.5599 +/- 0.0338** | **0.8492 +/- 0.0155** | **0.5747 +/- 0.0423** | 0.6543 +/- 0.0501 | **0.4908 +/- 0.0352** | **0.8050 +/- 0.0170** |
| **HistGradientBoosting** | 0.5473 +/- 0.0216 | 0.8386 +/- 0.0181 | 0.5574 +/- 0.0342 | 0.6419 +/- 0.0283 | 0.4779 +/- 0.0273 | 0.7984 +/- 0.0140 |

*Note: All candidate models utilize balanced class weighting and reproducible seed (random_state = 42). Bold values indicate the superior mean score in each metric column.*

Figure 7 illustrates the comparative performance across all six validation metrics.

![Figure 7](figures/figure_01_model_cv_comparison.png)

**Figure 7.** 10-Fold Stratified Cross-Validation Benchmark

### Key Performance Insights:
- **Baseline Trade-off:** Logistic Regression achieved high recall (78.69%) but low precision (40.21%), generating excessive false positive classifications. Its overall F1-score (0.5316) was the lowest among candidate models.
- **Ensemble Superiority:** Random Forest achieved the highest F1-score (0.5599), ROC-AUC (0.8492), PR-AUC (0.5747), and overall accuracy (80.50%). Its precision (49.08%) exceeded the baseline by nearly 9 percentage points.
- **Boosting Comparison:** HistGradientBoosting performed competitively (F1 = 0.5473, ROC-AUC = 0.8386) but did not surpass Random Forest in any key metric.

---

## Statistical Comparison

To test whether Random Forest's higher F1-score was statistically significant or merely due to random split variation, we evaluated the differences across the 10 paired validation folds (Table 6).

**Table 6.** Statistical Significance Testing on 10-Fold Cross-Validation F1-Scores

| Comparison | Mean F1 Diff | Standard Paired t (p-value) | Nadeau & Bengio Corrected t (p-value) | Statistical Decision (alpha = 0.05) |
|---|---|---|---|---|
| **Random Forest vs. Logistic Regression** | +0.0283 | t = 3.3840 (**p = 0.0081**) | t = 2.3290 (**p = 0.0448**) | **Statistically Significant (p < 0.05)** |
| **Random Forest vs. HistGradientBoosting** | +0.0126 | t = 1.4398 (p = 0.1838) | t = 0.9908 (p = 0.3477) | Not Significant at alpha = 0.05 |

### Statistical Findings:
1. **Comparison with Linear Baseline:** The standard paired t-test between Random Forest and Logistic Regression yields $t = 3.3840$ ($p = 0.0081$). Because standard paired tests on cross-validation folds can overstate significance due to overlapping training folds, we also calculated the Nadeau and Bengio (2003) corrected resampled t-test, which incorporates the variance correction factor $(1/k + n_{\text{test}}/n_{\text{train}} = 1/10 + 1/9 \approx 0.2111)$. Under this conservative test, the improvement remains statistically significant ($t = 2.3290, p = 0.0448 < 0.05$). We reject the null hypothesis of equal performance at the 95% confidence level, confirming that Random Forest provides a genuine improvement over the linear baseline.
2. **Comparison with Boosting:** The paired t-test between Random Forest and HistGradientBoosting yields $t = 1.4398$ ($p = 0.1838$; Nadeau & Bengio corrected $t = 0.9908, p = 0.3477$). While this difference is not statistically significant at $\alpha = 0.05$, Random Forest consistently achieved a higher mean F1-score (+0.0126), higher ROC-AUC (+0.0106), higher PR-AUC (+0.0173), and better precision (+0.0129).

---

## Hyperparameter Tuning

We ran a grid search over key hyperparameters on the training set using 10-fold cross-validation. Table 7 presents the top five parameter configurations ranked by validation F1-score.

**Table 7.** Hyperparameter Sensitivity Analysis for Random Forest (10-Fold CV F1-Score)

| Rank | Max Depth | Min Leaf Size | Trees (Estimators) | Validation F1 (Mean +/- Std) | Training F1 (Mean) |
|---|---|---|---|---|---|
| **1** | **15** | **2** | **300** | **0.5599 +/- 0.0338** | 0.9372 |
| 2 | 20 | 2 | 300 | 0.5584 +/- 0.0346 | 0.9615 |
| 3 | 15 | 1 | 300 | 0.5578 +/- 0.0321 | 0.9824 |
| 4 | 20 | 1 | 300 | 0.5562 +/- 0.0354 | 0.9994 |
| 5 | 10 | 2 | 300 | 0.5512 +/- 0.0284 | 0.8241 |

### Parameter Selection Rationale:
- **Tree Depth (`max_depth = 15`):** A maximum depth of 15 captures non-linear chemical thresholds without letting individual trees memorize noise. Restricting depth to 10 resulted in underfitting (F1 = 0.5512), while depth 20 raised training F1 (0.9615) but slightly lowered validation F1 (0.5584), showing early signs of overfitting.
- **Leaf Regularization (`min_samples_leaf = 2`):** Requiring at least 2 samples per leaf prevents terminal nodes from fitting to single noisy samples, stabilizing validation performance.
- **Ensemble Scale (`n_estimators = 300`):** An ensemble of 300 trees provides sufficient variance reduction, stabilizing probability estimates and feature importance rankings.

---

## Final Model Choice

Based on cross-validation results, statistical hypothesis testing, and hyperparameter tuning, we selected the **Random Forest Classifier** (`n_estimators = 300, max_depth = 15, min_samples_split = 4, min_samples_leaf = 2, class_weight = 'balanced'`) as our final model. It provides:
1. Statistically significant outperformance over the linear baseline ($p = 0.0081$ uncorrected, $p = 0.0448$ Nadeau & Bengio corrected).
2. The highest validation F1-score (0.5599), ROC-AUC (0.8492), and PR-AUC (0.5747).
3. Inherent invariance to monotonic transformations and outliers.
4. Reliable rank-order discrimination suitable for decision threshold tuning.

Section 5 presents the out-of-sample evaluation and diagnostic analysis of this model on the held-out test set.

---

# Model Analysis

We evaluated our final Random Forest model on the 1,064 unseen wines in `data/test.csv`. This section examines out-of-sample test performance, error trade-offs, diagnostic curves, which chemical features most strongly drove predictions, and how the model performed on red versus white wines.

---

## Test Set Performance

Table 8 summarizes the final performance metrics evaluated on the 1,064 held-out test observations.

**Table 8.** Final Random Forest Evaluation on Held-Out Test Dataset (N = 1,064)

| Performance Metric | Test Score | Operational Interpretation |
|---|---|---|
| **Accuracy** | **83.08%** (0.8308) | Correctly classifies 884 of 1,064 total test wines |
| **Precision** | **54.10%** (0.5410) | 145 of 268 wines flagged as high quality are true premium selections |
| **Recall (Sensitivity)** | **71.78%** (0.7178) | Successfully identifies 145 of the 202 true high-quality test wines |
| **Specificity** | **85.73%** (0.8573) | Correctly identifies 739 of 862 ordinary wines as ordinary |
| **F1-Score** | **0.6170** | Robust harmonic balance between precision and sensitivity |
| **ROC-AUC** | **0.8738** | Strong global discriminability across all operating thresholds |
| **PR-AUC (Average Precision)** | **0.6230** | Substantial minority-class enrichment (+43.32% over 18.98% baseline) |
| **Brier Score Loss** | **0.1225** | Lower than the 0.1538 baseline of always predicting the 19% base rate; see the calibration note below |

Table 9 presents the detailed classification report for both target classes.

**Table 9.** Test Classification Report by Class

| Class Label | Support (N) | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Ordinary Wine (`is_good = 0`)** | 862 (81.02%) | 0.9284 | 0.8573 | 0.8914 |
| **High-Quality Wine (`is_good = 1`)** | 202 (18.98%) | 0.5410 | 0.7178 | 0.6170 |
| **Macro Average** | 1,064 | 0.7347 | 0.7876 | 0.7542 |
| **Weighted Average** | 1,064 | 0.8549 | 0.8308 | 0.8393 |

The test set results demonstrate strong generalization. Test F1-score (0.6170) and test ROC-AUC (0.8738) exceed the 10-fold cross-validation estimates (0.5599 and 0.8492, respectively), confirming that the model did not suffer from overfitting during cross-validation.

---

## Confusion Matrix and Decision Trade-offs

Figure 8 displays the raw count and normalized confusion matrices on the test dataset.

![Figure 8](figures/figure_02_confusion_matrix.png)

**Figure 8.** Confusion Matrix Analysis

Out of 1,064 test evaluations:
- **True Negatives (TN):** 739 samples (69.45% of total, 85.73% of ordinary wines).
- **True Positives (TP):** 145 samples (13.63% of total, 71.78% of high-quality wines).
- **False Positives (FP: Type I Error):** 123 samples (11.56% of total, 14.27% of ordinary wines).
- **False Negatives (FN: Type II Error):** 57 samples (5.36% of total, 28.22% of high-quality wines).

### Practical Implications of Errors:

1. **Type I Errors (False Positives: 123 wines):**
   - *Operational Consequence:* An ordinary wine is labeled as high quality and could be routed to a premium bottling line.
   - *Risk:* Subpar wine on store shelves under a reserve label damages consumer trust and brand reputation.
   - *Mitigation:* The model can serve as a pre-screening filter, with flagged wines confirmed by human sensory evaluation before bottling.
2. **Type II Errors (False Negatives: 57 wines):**
   - *Operational Consequence:* A high-quality wine is labeled ordinary and sold in bulk or standard distribution.
   - *Risk:* Forfeited margin. Premium wine sold as table wine loses potential revenue.
   - *Mitigation:* The balanced class weighting keeps false negatives low (only 5.36% of total samples), catching over 71.7% of all premium lots.

---

## ROC, Precision-Recall, and Calibration Curves

Figure 9 illustrates the Receiver Operating Characteristic (ROC) curve alongside the Precision-Recall (PR) curve for the held-out test dataset.

![Figure 9](figures/figure_03_roc_and_pr_curves.png)

**Figure 9.** ROC and Precision-Recall Curves

### Key Diagnostic Curve Observations:
- **ROC Trajectory (AUC = 0.8738):** The ROC curve ascends steeply in the low false positive rate region (0.00 to 0.20), reaching over 70% recall while keeping the false positive rate below 15%. This demonstrates strong discriminative power across wide operating conditions.
- **Precision-Recall Dynamics (PR-AUC = 0.6230):** In imbalanced domains, the PR curve provides an informative assessment of minority class precision. The horizontal reference line at 0.1898 represents a random guess baseline. The Random Forest model maintains a substantial elevation above this baseline across the entire recall spectrum.
- **Adjustable Decision Thresholds:** The default 0.50 probability threshold yields 54.10% precision at 71.78% recall (F1 = 0.6170). Depending on commercial strategy, the operating threshold can be adjusted along the precision-recall trade-off continuum:
  - *Brand Protection Policy (High Precision):* Elevating the classification threshold to 0.65 increases precision to 62.86% (at 43.56% recall), while setting the threshold to 0.70 yields 68.93% precision (at 35.15% recall) and 0.75 yields 73.68% precision (at 27.72% recall), ensuring that only high-confidence lots receive premium bottling.
  - *Volume Harvesting Policy (High Recall):* Lowering the threshold to 0.40 captures 80.69% of true high-quality wines (at 45.79% precision), while setting the threshold to 0.35 captures 85.15% of high-quality wines (at 41.95% precision) for secondary sommelier auditing.

Figure 10 displays the probability calibration curve (reliability diagram).

![Figure 10](figures/figure_06_calibration_curve.png)

**Figure 10.** Probability Calibration Reliability Curve

The calibration curve illustrates a characteristic upward probability shift: across the lower and middle probability intervals, predicted probabilities exceed empirical event frequencies (for example, wines assigned a model probability of ~0.45 exhibit an actual high-quality rate of 20.45%). This probability distortion is a direct mathematical consequence of cost-sensitive learning (`class_weight='balanced'`), which artificially scales minority class misclassification loss by a factor of 2.64 during tree splitting. While this reweighting significantly improves minority class discrimination and decision-boundary separability (achieving a test ROC-AUC of 0.8738 and F1-score of 0.6170), raw model probabilities should be interpreted as relative risk indices rather than calibrated probabilities unless post-hoc calibration (such as Platt scaling or isotonic regression) is applied. Despite this systematic shift, the overall test Brier score loss is 0.1225 (substantially outperforming a naive prevalence baseline of 0.1538).

---

## Feature Importance

To understand the chemical drivers governing wine quality classification, we analyzed feature importance using two distinct analytical methods:
1. **Gini Impurity Importance (Mean Decrease in Impurity, MDI):** Total impurity reduction across all 300 bagged trees during training.
2. **Permutation Feature Importance:** Model-agnostic drop in test F1-score when each feature is randomly shuffled across 20 iterations on `data/test.csv`.

Table 10 compares the rankings from both methodologies.

**Table 10.** Physicochemical Feature Importance Comparison

| Feature Name | Permutation Mean (F1 Drop) | Permutation Std | Gini Impurity (MDI) | Physical/Chemical Mechanism |
|---|---|---|---|---|
| **Alcohol** | **0.2461** | 0.0268 | **0.2251** | Ethanol content: body, perceived sweetness, aromatic volatility |
| **Chlorides** | 0.0796 | 0.0167 | 0.0869 | Salt content: mineral mouthfeel; excess imparts saltiness and bitterness |
| **Density** | 0.0768 | 0.0151 | 0.1235 | Inversely related to alcohol and sugar: overall mouthfeel and body |
| **Total Sulfur Dioxide** | 0.0549 | 0.0177 | 0.0799 | Preservative: antioxidant/antimicrobial; excess dulls fruit flavors |
| **Volatile Acidity** | 0.0437 | 0.0146 | 0.0858 | Acetic acid: vinegar fault; lower levels required for clean sensory profile |
| **Sulphates** | 0.0432 | 0.0093 | 0.0696 | Potassium sulphate: antimicrobial synergy, fruit freshness enhancer |
| **Free Sulfur Dioxide** | 0.0377 | 0.0113 | 0.0652 | Active unbound preservative protecting wine from oxidation |
| **Citric Acid** | 0.0367 | 0.0137 | 0.0738 | Minor organic acid: provides crispness and fresh citrus notes |
| **pH** | 0.0323 | 0.0085 | 0.0613 | Acidity buffer: microbial stability and perception of tartness |
| **Residual Sugar** | 0.0256 | 0.0082 | 0.0678 | Unfermented sugar: balance against high acidity in Vinho Verde |
| **Fixed Acidity** | 0.0250 | 0.0072 | 0.0576 | Tartaric and malic acids: foundational crisp structural acidity |
| **is_red** | 0.0044 | 0.0040 | 0.0035 | Variety indicator: secondary after direct chemical measurements |

Figure 11 illustrates the comparative importance profiles.

![Figure 11](figures/figure_04_feature_importance.png)

**Figure 11.** Physicochemical Feature Importance

### Key Chemical Insights:
1. **Alcohol is the Dominant Quality Predictor:** Shuffling alcohol on the test set drops the model's F1-score by 0.2461 (from 0.6170 down to ~0.3709). Gini importance confirms this finding (0.2251). In cool maritime climates like Vinho Verde, higher natural alcohol indicates fully ripe grapes with concentrated flavors, balanced acids, and proper phenolic maturity.
2. **Chlorides and Density Form Key Structural Markers:** Chlorides (salt content) and density represent the second and third most impactful features in permutation testing. Excessive chlorides impart an unpleasant saline astringency that judges penalize heavily. Density integrates alcohol and sugar levels, serving as a composite indicator of extraction and body.
3. **Acid Balance and Fermentation Cleanliness:** Volatile acidity (acetic acid) is a primary marker of bacterial spoilage. High-quality wines consistently maintain low volatile acidity. Sulphates and sulfur dioxide provide microbial protection, ensuring fruit notes remain clean and stable.
4. **Wine Type (`is_red`) Has Minor Direct Impact:** The binary type indicator shows low permutation importance (0.0044). Once specific chemical concentrations (acidity, density, and sulfur levels) are known, the explicit red vs. white label provides little additional predictive information.

---

## Red vs. White Wine Performance

To evaluate whether the model functions equally well across wine varieties, we disaggregated test set performance by wine type.

**Table 11.** Subgroup Disaggregation on Test Dataset

| Wine Variety | Samples (N) | Good Wines (N) | Prevalence | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Red Wine** | 275 | 37 | 13.45% | **90.18%** | **60.42%** | **78.38%** | **0.6824** | **0.9307** | **0.7255** |
| **White Wine** | 789 | 165 | 20.91% | 80.61% | 52.73% | 70.30% | 0.6026 | 0.8522 | 0.6044 |

Figure 12 shows the disaggregated ROC and PR curves for red and white wines.

![Figure 12](figures/figure_05_subgroup_analysis.png)

**Figure 12.** Subgroup Performance Comparison

### Subgroup Findings:
- **Red Wine Results:** The model achieves higher performance on red wines: 90.18% accuracy, 78.38% recall, 0.6824 F1-score, and an ROC-AUC of 0.9307 (with PR-AUC of 0.7255 against a 13.45% baseline). In red wines, high quality is strongly demarcated by low volatile acidity (avoiding vinegar spoilage) and high sulphates.
- **White Wine Results:** White wines exhibit an ROC-AUC of 0.8522 and an F1-score of 0.6026. White wines are more heterogeneous in residual sugar and acidity balance (ranging from dry to off-dry), creating more overlap between ordinary and high-quality profiles.
- **Takeaway:** The model provides strong classification utility across both styles, with especially high discrimination on red wines.

---

## Model Validity and Overfitting Check

To verify algorithmic stability and confirm that the model did not memorize training artifacts, we compared performance across training, 10-fold cross-validation, and held-out testing partitions.

**Table 12.** Performance Comparison across Training, Cross-Validation, and Test Sets

| Metric | Training Set (Fit, N = 4,256) | 10-Fold CV Validation (Mean +/- Std) | Held-Out Test Set (Unseen, N = 1,064) |
|---|---|---|---|
| **Accuracy** | 97.46% | 80.50% +/- 1.70% | 83.08% |
| **F1-Score** | 0.9372 | 0.5599 +/- 0.0338 | 0.6170 |
| **ROC-AUC** | 0.9991 | 0.8492 +/- 0.0155 | 0.8738 |
| **Precision** | 0.8924 | 0.4908 +/- 0.0352 | 0.5410 |
| **Recall** | 0.9864 | 0.6543 +/- 0.0501 | 0.7178 |

### Findings:
1. **Regularization Effectiveness:** While tree ensembles naturally exhibit high training fit due to bootstrap sample isolation, the validation metrics across 10 folds remained stable with low variance ($\sigma_{\text{ROC-AUC}} = 0.0155$, $\sigma_{\text{F1}} = 0.0338$).
2. **Generalization Confirmation:** The test set scores (F1 = 0.6170, ROC-AUC = 0.8738) closely mirror and slightly exceed the cross-validation estimates. This confirms that the hyperparameter constraints (`max_depth = 15, min_samples_leaf = 2`) prevented the ensemble from overfitting.

---

## Limitations

While the Random Forest classifier demonstrates strong empirical performance, practical deployment in commercial viticulture must account for several structural limitations:

1. **Regional viticultural specificity:** All samples originate from the Vinho Verde region in northwest Portugal. Vinho Verde is characterized by a cool, humid Atlantic climate producing wines with high natural acidity, lower alcohol levels, and distinctive carbonation. The physicochemical boundaries learned by this model may not generalize to warmer viticultural regions (such as Napa Valley, Rioja, or Barossa Valley) where baseline alcohol, sugar, and tannin profiles differ markedly.
2. **Subjectivity in sensory quality ratings:** The ground-truth quality score represents the median rating from at least three sensory experts on a 0 to 10 scale. Expert wine tasting involves subjective palate differences, fatigue during lengthy tasting flights, and ambient temperature sensitivity. This sensory variance introduces irreducible label noise near the decision boundary (qualities 6 vs. 7).
3. **Absence of commercial and viticultural variables:** The dataset contains purely post-fermentation laboratory chemistry. It omits critical factors that heavily drive commercial market pricing, including:
   - Specific grape cultivar (e.g., Alvarinho, Loureiro, Vinhão).
   - Vintage year and meteorological growing season conditions.
   - Oak barrel maturation duration and toast profiles.
   - Phenolic and anthocyanin compounds (tannins, color pigments).
   - Brand equity, packaging, and retail distribution channels.

---

# Conclusion and Recommendations

## Summary

We set out to answer whether a wine's lab measurements can predict whether experts will rate it as good (a score of 7 or higher). Using 5,320 unique red and white Vinho Verde wines, we compared three classification models: logistic regression, random forest, and gradient boosting. Our final model, a random forest, achieved a ROC-AUC of 0.874 and an F1 score of 0.617 on 1,064 wines it had never seen. At the default threshold it found 72% of the good wines (145 of 202), and 54% of the wines it labeled good really were good.

The random forest was significantly better than logistic regression, even after correcting the significance test for overlapping cross-validation folds (p = 0.045). It was not significantly better than gradient boosting, so the two ensemble models perform about the same on this data. The test scores closely matched the cross-validation scores, so the model was not overfit to the training data.

The answer to our question is a qualified yes. Chemistry carries a strong signal about quality, but not enough to replace expert tasters. Further tuning is unlikely to change this much: the default threshold of 0.50 is within 0.003 of the best possible F1 score, and adding engineered features such as chemical ratios did not improve cross-validation performance. The remaining errors likely come partly from the subjectivity of expert scores, especially for wines near the cutoff between 6 and 7.

## What Drives Quality

- **Alcohol content matters most.** Good wines have a median alcohol content of 11.6%, compared with 10.1% for other wines. Shuffling alcohol on the test set lowered the model's F1 score by 0.246, about three times more than any other feature.
- **Chlorides and density come next.** Lower chlorides (salt content) and lower density are associated with good wines. Density is closely tied to alcohol and sugar, so part of its importance reflects alcohol content.
- **Low volatile acidity marks a clean wine.** Volatile acidity is linked to a vinegar-like taste, and good wines consistently have lower levels.
- **Wine type matters little once the chemistry is known.** Red and white wines have very different chemistry, but the red or white label itself added almost nothing to the predictions. The model performed better on red wines (ROC-AUC 0.931 vs. 0.852), although the test set contained only 37 good red wines, so this result is less certain.

These are associations in the data, not proven causes. Raising a wine's alcohol content, for example, will not necessarily make experts rate it higher.

## Recommendations

1. **Use the model as a first screening step, not a replacement for tasters.** Run every batch's lab results through the model and send the wines it flags to expert tasting. At the default threshold, the model flagged 268 of 1,064 test wines (25%), which would cut routine tasting volume by about 75%. The trade-off is that it missed 57 of the 202 good wines (28%), which would not be tasted under this policy.
2. **Choose the decision threshold based on business priorities.** If missing a good wine is costly, lower the threshold: at 0.35, the model finds 85% of good wines, but only 42% of the flagged wines are good. If tasting capacity or brand reputation matters most, raise it: at 0.70, 69% of flagged wines are good, but the model finds only 35% of them. The threshold should be chosen with cross-validation on the training data, not on the test set.
3. **Treat the model's scores as rankings, not exact probabilities.** Because the model was trained with balanced class weights, its predicted probabilities are higher than the true rates. A score of 0.45 corresponds to a real good-wine rate of about 20%. If the business needs true probabilities, the model should be recalibrated first.
4. **Monitor alcohol, volatile acidity, and chlorides during production.** These are the measurements most closely linked to high ratings, and batches with unusually high volatile acidity or chlorides can be flagged for review early.
5. **Test the model before using it on other wines.** It was trained only on Vinho Verde wines and should be checked on a sample of wines with known expert scores before being used for other regions or styles.

## Future Work

- **More and broader data:** wines from other regions and grape varieties, plus information the dataset lacks, such as grape variety, vintage, and production methods.
- **Predicting the score itself:** treating quality as an ordered scale instead of a yes/no label would avoid the sharp cutoff between 6 and 7.
- **Separate models for red and white wines,** if more red wine data becomes available.
- **Probability calibration and cost-based thresholds:** recalibrating the model's scores (for example with Platt scaling or isotonic regression), and choosing a threshold that minimizes a winery's actual costs.

---

# References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/186/wine+quality

Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

The pandas development team. (2020). *pandas-dev/pandas: Pandas*. Zenodo. https://doi.org/10.5281/zenodo.3509134

Waskom, M. L. (2021). seaborn: Statistical data visualization. *Journal of Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021

---

# Appendix: Technical Notebooks

The code and output of our three Jupyter notebooks, in the order they are run. The PDF version of this report includes their full output.

- Appendix A.1: [01_data_cleaning.ipynb](../notebooks/01_data_cleaning.ipynb)
- Appendix A.2: [02_eda.ipynb](../notebooks/02_eda.ipynb)
- Appendix A.3: [03_modeling.ipynb](../notebooks/03_modeling.ipynb)
