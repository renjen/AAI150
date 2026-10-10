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
