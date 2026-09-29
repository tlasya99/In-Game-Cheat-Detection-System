# In-Game Cheat Detection

A coursework prototype that scores gameplay sessions for moderator review.

## Run locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the dashboard:

```powershell
python -m streamlit run moderator_dashboard.py
```

Retrain the baseline model:

```powershell
python train_model.py
```

The dashboard uses `final_dataset.csv` and `cheat_detection_model.joblib`. It saves moderator decisions in `review_actions.csv` when reviewers submit them.

## Model and data limitations

The supplied data is synthetic. The reported metrics are preliminary and do not establish real world accuracy. A model score is a review signal, not proof of cheating. This prototype does not include authentication or role based access. Local CSV review storage may not persist in hosted environments.
