# AAI150

Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality

We're predicting if a wine is good (quality 7 or higher) or not good based on its chemical stuff, so it's a classification problem.

## What's in here

- `data/` has the original red and white wine csvs from UCI, plus the cleaned files:
  - `wine_clean.csv` is the full cleaned data, use this for EDA
  - `train.csv` and `test.csv` are the 80/20 split, use these for the models
- `notebooks/01_data_cleaning.ipynb` is the cleaning notebook (Person A)
- `requirements.txt` has the packages we need

## Setup

You need Python (3.10+) and Git installed. The data's already in the repo so you don't have to download anything.

Clone it:

```
git clone https://github.com/renjen/AAI150.git
cd AAI150
```

Make a virtual environment and turn it on.

Mac:
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` in your terminal once it's on. You have to run the activate line again whenever you open a new terminal.

Install the packages:

```
pip install -r requirements.txt
```

To open the notebooks in VS Code, get the Python and Jupyter extensions, open a notebook, hit "Select Kernel" in the top right and pick `.venv`. Or you can just run `jupyter notebook` and use it in the browser.

To make sure it all works, open `01_data_cleaning.ipynb` and do Run All. It should run with no errors and remake the cleaned csvs in `data/`.

## Git workflow

- `git pull` before you start working so you have the latest stuff
- make your own branch, like `git checkout -b person-b-eda`
- commit your work and push your branch (`git push -u origin person-b-eda`)
- open a pull request on GitHub and have someone else review it before it goes into main

How the work is split up: 

The dataset itself: ~6,500 rows total (1,599 red + 4,898 white), 11 numeric input features (acidity, sugar, chlorides, sulfur dioxide, density, pH, sulphates, alcohol), and a quality score (0–10, though in practice mostly 3–9) as the target. That target can be treated as a regression problem (predict the score) or turned into classification (e.g., "good" vs. "not good" wine). Deciding that early actually shapes how you split the work, so it's worth a quick team decision.
Split by project phase, not by person-does-their-own-thing. The reason "equal split" trips people up is that if one person owns cleaning, one owns EDA, one owns modeling in total isolation, you end up with three people who each understand only a third of the project — bad for the report's coherence and bad if your instructor asks anyone a question about a part they didn't touch. Instead, each phase gets a driver who does the bulk of the work and writes that report section, but the code gets reviewed by the other two (which also satisfies the syllabus's "everyone needs to code and review the code" line).
Concretely, with three people:
Person A — Data Cleaning & Preparation. Loads red and white wine files, decides whether to combine them (with a "type" indicator column) or treat separately, checks for missing values/outliers/duplicates, checks data types, maybe does a train/test split. Writes the "Data Cleaning/Preparation" report section.
Person B — Exploratory Data Analysis. Summary statistics, distributions of each feature, correlation matrix/heatmap, looks at how features relate to quality score, visualizes class imbalance if you go classification. Writes the "EDA" section.
Person C — Model Selection & Model Analysis. Fits a couple of candidate models (e.g., linear/logistic regression as a baseline, plus something like random forest or gradient boosting), compares performance, picks a final model, evaluates it (residuals if regression, confusion matrix/ROC if classification). Writes "Model Selection" and "Model Analysis."
Together: Introduction and Conclusion/Recommendations get written jointly at the end, once everyone knows what actually happened — you can't write a real intro or conclusion before the analysis exists. The video presentation and the appendix notebook are also shared: literally everyone should be able to explain any part of the pipeline for the video, and the notebook should read as one continuous story (Person A's cleaning code feeding into Person B's EDA feeding into Person C's models), not three stapled-together scripts.
Cadence that keeps this from collapsing in week 7: each phase should have a hard internal deadline before Module 7, with a short check-in when a phase hands off to the next person (e.g., Person A tells Person B "here's the cleaned dataframe, here's what I did to it" before EDA starts). A shared GitHub repo with a branch per person and pull requests for review makes the "everyone reviews code" requirement visible and easy to prove.