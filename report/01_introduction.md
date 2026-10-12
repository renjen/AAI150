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
