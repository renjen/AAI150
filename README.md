# AAI150

https://archive.ics.uci.edu/dataset/186/wine+quality 




How the work is split up: 

The dataset itself: ~6,500 rows total (1,599 red + 4,898 white), 11 numeric input features (acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol), and a quality score (0–10, though in practice mostly 3–9) as the target. That target can be treated as a regression problem (predict the score) or turned into classification (e.g., "good" vs. "not good" wine). Deciding that early actually shapes how you split the work, so it's worth a quick team decision.
Split by project phase, not by person-does-their-own-thing. The reason "equal split" trips people up is that if one person owns cleaning, one owns EDA, one owns modeling in total isolation, you end up with three people who each understand only a third of the project — bad for the report's coherence and bad if your instructor asks anyone a question about a part they didn't touch. Instead, each phase gets a driver who does the bulk of the work and writes that report section, but the code gets reviewed by the other two (which also satisfies the syllabus's "everyone needs to code and review the code" line).
Concretely, with three people:
Person A — Data Cleaning & Preparation. Loads red and white wine files, decides whether to combine them (with a "type" indicator column) or treat separately, checks for missing values/outliers/duplicates, checks data types, maybe does a train/test split. Writes the "Data Cleaning/Preparation" report section.
Person B — Exploratory Data Analysis. Summary statistics, distributions of each feature, correlation matrix/heatmap, looks at how features relate to quality score, visualizes class imbalance if you go classification. Writes the "EDA" section.
Person C — Model Selection & Model Analysis. Fits a couple of candidate models (e.g., linear/logistic regression as a baseline, plus something like random forest or gradient boosting), compares performance, picks a final model, evaluates it (residuals if regression, confusion matrix/ROC if classification). Writes "Model Selection" and "Model Analysis."
Together: Introduction and Conclusion/Recommendations get written jointly at the end, once everyone knows what actually happened — you can't write a real intro or conclusion before the analysis exists. The video presentation and the appendix notebook are also shared: literally everyone should be able to explain any part of the pipeline for the video, and the notebook should read as one continuous story (Person A's cleaning code feeding into Person B's EDA feeding into Person C's models), not three stapled-together scripts.
Cadence that keeps this from collapsing in week 7: each phase should have a hard internal deadline before Module 7, with a short check-in when a phase hands off to the next person (e.g., Person A tells Person B "here's the cleaned dataframe, here's what I did to it" before EDA starts). A shared GitHub repo with a branch per person and pull requests for review makes the "everyone reviews code" requirement visible and easy to prove.