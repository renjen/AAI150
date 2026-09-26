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
