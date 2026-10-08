# Model Selection

In this phase, we built and evaluated machine learning classifiers to predict whether a wine is high quality based on its 11 physicochemical measurements and wine type. Following the definition from data cleaning, a wine is good (`is_good = 1`) if its sensory evaluation score is 7 or higher, and ordinary (`is_good = 0`) if its score is 6 or lower.

The feature set includes 12 variables:
- 11 continuous chemical features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.
- 1 binary indicator: `is_red` (1 for red wine, 0 for white wine).

To prevent target leakage, the original continuous `quality` score and the text `type` column were dropped before modeling.

---

## 4.2 Preprocessing and Prevention of Data Leakage

In predictive modeling pipelines, data leakage occurs when information from outside the training fold is used to transform features or select parameters. To prevent leakage:

1. **Strict Partitioning:** The 80/20 stratified split generated during data preparation was strictly preserved. All model exploration, scaling, parameter estimation, and cross-validation were conducted exclusively on `train.csv` (N = 4,256). The test partition (`test.csv`, N = 1,064) remained unobserved until final out-of-sample evaluation.
2. **Encapsulated Preprocessing Pipelines:** For distance-sensitive algorithms requiring feature normalization (specifically Logistic Regression), standard z-score normalization (`StandardScaler`) was encapsulated within a scikit-learn `Pipeline`. During 10-fold cross-validation, the scaler was fitted solely on the 9 training folds in each split and then applied to transform the held-out validation fold. Tree-based ensemble models (Random Forest and Gradient Boosting) are invariant to monotonic feature scaling and operate directly on raw physicochemical units, preserving native numerical interpretability.

---

## 4.3 Handling Class Imbalance

As documented in data preparation, the target distribution is notably imbalanced:
- Ordinary Wines (`is_good = 0`): 81.04% of training observations (3,449 samples).
- High-Quality Wines (`is_good = 1`): 18.96% of training observations (807 samples).

A naive zero-rule classifier that assigns every wine to the majority class would achieve 81.04% accuracy while failing completely to detect a single high-quality wine. To solve this challenge without synthetically distorting the feature joint distributions through oversampling:

1. **Stratified Partitioning:** A 10-fold Stratified K-Fold cross-validation scheme (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`) was enforced across all model comparisons. This guarantees that each validation fold contains the exact 18.96% positive class proportion.
2. **Cost-Sensitive Weighting:** All candidate models incorporated cost-sensitive loss weighting (`class_weight='balanced'`). The loss associated with misclassifying a minority-class observation was scaled inversely to class frequency according to:

$$w_j = \frac{N}{2 \times N_j}$$

where $N$ is the total number of training samples and $N_j$ is the count of samples in class $j$. This assigns a weight of approximately 2.64 to high-quality wines and 0.62 to ordinary wines, penalizing false negatives and compelling candidate algorithms to balance sensitivity with specificity.

---

## 4.4 Evaluation Metric Protocol

Given class imbalance, classification accuracy is an uninformative and potentially deceptive performance metric. The modeling protocol evaluates models across five specialized performance dimensions:

1. **Precision:** $\frac{TP}{TP + FP}$. Reflects the proportion of predicted high-quality wines that are truly high quality. High precision protects brand reputation by preventing mediocre wines from being sold as premium reserves.
2. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$. Reflects the proportion of actual high-quality wines successfully identified by the algorithm. High recall minimizes forfeited revenue by preventing premium wines from being sold as standard table wine.
3. **F1-Score:** The harmonic mean of precision and recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$). F1-score serves as the primary optimization and selection criterion because it rewards models that balance precision and recall without sacrificing either dimension.
4. **ROC-AUC:** Area Under the Receiver Operating Characteristic curve. Measures global discriminability across all classification thresholds, independent of class prevalence.
5. **PR-AUC (Average Precision):** Area Under the Precision-Recall curve. In imbalanced settings, PR-AUC evaluates precision against recall across all thresholds, establishing a benchmark against the 0.1896 random baseline.

---

## 4.5 Candidate Model Architectures and Theoretical Rationale

Three distinct model families were benchmarked to identify the optimal balance between predictive capacity, resistance to overfitting, and operational interpretability:

1. **Logistic Regression (Parametric Baseline):**
   - *Rationale:* Provides a transparent, linear reference baseline. It models log-odds as a linear combination of normalized inputs with $L_2$ ridge regularization ($C = 1.0$) and balanced class weighting.
   - *Limitations:* Assumes linear decision boundaries and cannot capture complex chemical synergies (e.g., non-linear interactions between alcohol, sugar, and volatile acidity) without manual interaction terms.
2. **Random Forest Classifier (Ensemble Bagging: Primary Model):**
   - *Rationale:* An ensemble of 300 decorrelated decision trees built on bootstrap samples with random feature subsampling. Random forest handles skewed chemical distributions, resists univariate outliers, models high-order non-linear interactions, and mitigates individual tree variance through bagging.
   - *Regularization:* Constrained to a maximum tree depth of 15, minimum split size of 4, and minimum leaf size of 2 to inhibit memorization of noisy expert scores.
3. **Histogram-Based Gradient Boosting (Ensemble Boosting: Comparative Model):**
   - *Rationale:* An iterative sequential boosting architecture that bins continuous features into discrete 256-integer histograms, optimizing a logistic loss function through gradient descent.
   - *Configuration:* Evaluated with 200 boosting iterations, a learning rate of 0.08, maximum tree depth of 6, and balanced sample weighting.

---

## 4.6 10-Fold Stratified Cross-Validation Benchmark

Table 1 summarizes the cross-validation performance of all three candidate architectures across the 10 stratified validation folds. All metrics report the fold mean plus or minus one standard deviation.

**Table 1.** 10-Fold Stratified Cross-Validation Benchmark on Training Dataset (N = 4,256)

| Model Architecture | F1-Score | ROC-AUC | PR-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.5316 +/- 0.0263 | 0.8306 +/- 0.0195 | 0.5282 +/- 0.0461 | **0.7869 +/- 0.0250** | 0.4021 +/- 0.0282 | 0.7361 +/- 0.0251 |
| **Random Forest (Final Model)** | **0.5599 +/- 0.0338** | **0.8492 +/- 0.0155** | **0.5747 +/- 0.0423** | 0.6543 +/- 0.0501 | **0.4908 +/- 0.0352** | **0.8050 +/- 0.0170** |
| **HistGradientBoosting** | 0.5473 +/- 0.0216 | 0.8386 +/- 0.0181 | 0.5574 +/- 0.0342 | 0.6419 +/- 0.0283 | 0.4779 +/- 0.0273 | 0.7984 +/- 0.0140 |

*Note: All candidate models utilize balanced class weighting and reproducible seed (random_state = 42). Bold values indicate the superior mean score in each metric column.*

Figure 1 illustrates the comparative performance across all six validation metrics.

![Figure 1: 10-Fold Stratified Cross-Validation Benchmark](figures/figure_01_model_cv_comparison.png)

### Key Performance Insights:
- **Baseline Trade-off:** Logistic Regression achieved high recall (78.69%) but poor precision (40.21%), generating excessive false positive classifications. Its overall F1-score (0.5316) was the lowest among candidate models.
- **Ensemble Superiority:** Random Forest achieved the highest F1-score (0.5599), the highest ROC-AUC (0.8492), the highest PR-AUC (0.5747), and the highest overall accuracy (80.50%). Its precision (49.08%) exceeded the baseline by nearly 9 percentage points.
- **Boosting Comparison:** HistGradientBoosting performed competitively (F1 = 0.5473, ROC-AUC = 0.8386) but did not surpass Random Forest in any key metric.

---

## 4.7 Statistical Significance: Paired Student's t-Test

To verify that Random Forest's superior performance was statistically meaningful and not an artifact of data partitioning, we performed two-tailed paired Student's t-tests across the 10 paired validation fold F1-scores.

**Table 2.** Paired Student's t-Test on 10-Fold Cross-Validation F1-Scores

| Comparison | Mean Fold F1 Difference | t-Statistic | p-Value | Statistical Decision (alpha = 0.05) |
|---|---|---|---|---|
**Table 2.** Statistical Significance Testing on 10-Fold Cross-Validation F1-Scores

| Comparison | Mean F1 Diff | Standard Paired t (p-value) | Nadeau & Bengio Corrected t (p-value) | Statistical Decision (alpha = 0.05) |
|---|---|---|---|---|
| **Random Forest vs. Logistic Regression** | +0.0283 | t = 3.3840 (**p = 0.0081**) | t = 2.3290 (**p = 0.0448**) | **Statistically Significant (p < 0.05)** |
| **Random Forest vs. HistGradientBoosting** | +0.0126 | t = 1.4398 (p = 0.1838) | t = 0.9908 (p = 0.3477) | Not Significant at alpha = 0.05 |

### Statistical Findings:
1. **Confirmation over Linear Baseline:** The standard paired t-test between Random Forest and Logistic Regression yields $t = 3.3840$ ($p = 0.0081$). Because standard paired tests on cross-validation folds can underestimate variance due to overlapping training sets, we also applied the Nadeau and Bengio (2003) corrected resampled t-test, which incorporates the variance correction factor $(1/k + n_{\text{test}}/n_{\text{train}} = 1/10 + 1/9 \approx 0.2111)$. Under this conservative test, the comparison remains statistically significant ($t = 2.3290, p = 0.0448 < 0.05$). We reject the null hypothesis of equal performance at the 95% confidence level, demonstrating that Random Forest provides a statistically significant improvement over linear modeling.
2. **Ensemble Comparison:** The paired t-test between Random Forest and HistGradientBoosting yields $t = 1.4398$ ($p = 0.1838$; Nadeau & Bengio corrected $t = 0.9908, p = 0.3477$). While the difference between the two ensemble architectures does not achieve statistical significance at $\alpha = 0.05$, Random Forest consistently exhibits a higher mean F1-score (+0.0126), higher ROC-AUC (+0.0106), higher PR-AUC (+0.0173), and superior precision (+0.0129).

---

## 4.8 Hyperparameter Tuning and Sensitivity Analysis

A grid search was executed over structural hyperparameter combinations on the training set using 10-fold cross-validation. Table 3 presents the top five parameter configurations ranked by validation F1-score.

**Table 3.** Hyperparameter Sensitivity Analysis for Random Forest (10-Fold CV F1-Score)

| Rank | Max Depth | Min Leaf Size | Trees (Estimators) | Validation F1 (Mean +/- Std) | Training F1 (Mean) |
|---|---|---|---|---|---|
| **1** | **15** | **2** | **300** | **0.5599 +/- 0.0338** | 0.9372 |
| 2 | 20 | 2 | 300 | 0.5584 +/- 0.0346 | 0.9615 |
| 3 | 15 | 1 | 300 | 0.5578 +/- 0.0321 | 0.9824 |
| 4 | 20 | 1 | 300 | 0.5562 +/- 0.0354 | 0.9994 |
| 5 | 10 | 2 | 300 | 0.5512 +/- 0.0284 | 0.8241 |

### Parameter Selection Rationale:
- **Tree Depth (`max_depth = 15`):** A maximum depth of 15 allows the ensemble to capture non-linear physicochemical thresholds while preventing individual trees from growing to unconstrained leaf depths. Restricting depth to 10 resulted in underfitting (F1 = 0.5512), whereas depth 20 marginally increased training F1 (0.9615) while decreasing validation F1 (0.5584), signaling early overfitting.
- **Leaf Regularization (`min_samples_leaf = 2`):** Setting `min_samples_leaf = 2` ensures terminal leaf nodes cannot be formed on single isolated noise points, improving validation stability.
- **Ensemble Scale (`n_estimators = 300`):** An ensemble size of 300 trees provides sufficient variance reduction, stabilizing probability estimates and feature importance rankings.

---

## 4.9 Final Model Selection

Based on empirical validation, statistical hypothesis testing, and hyperparameter sensitivity analysis, the **Random Forest Classifier** (`n_estimators = 300, max_depth = 15, min_samples_split = 4, min_samples_leaf = 2, class_weight = 'balanced'`) is selected as the final production model. It provides:
1. Statistically significant outperformance over linear baselines ($p = 0.0081$ uncorrected, $p = 0.0448$ Nadeau & Bengio corrected).
2. The highest validation F1-score (0.5599), ROC-AUC (0.8492), and PR-AUC (0.5747).
3. Inherent invariance to monotonic transformations and extreme outliers.
4. Strong rank-order discrimination suitable for decision threshold adjustment in commercial winemaking applications.

Section 5 presents the complete out-of-sample evaluation and diagnostic analysis of this selected model on the held-out test dataset.
