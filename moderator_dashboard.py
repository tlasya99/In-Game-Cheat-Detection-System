from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "final_dataset.csv"
MODEL_PATH = ROOT / "cheat_detection_model.joblib"

st.set_page_config(page_title="Cheat Risk Predictor", page_icon="🎮", layout="wide")


@st.cache_data
def load_scored_sessions(data_path: str, model_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    missing_features = sorted(set(features) - set(data.columns))
    if missing_features:
        raise ValueError(f"Dataset is missing model features: {missing_features}")

    scores = model.predict_proba(data[features])[:, 1]
    rows = pd.DataFrame({
        "uuid": data["uuid"].astype(str),
        "risk_score": scores,
        "event_time": pd.to_numeric(data["time"], errors="coerce"),
    })
    sessions = rows.groupby("uuid", as_index=False).agg(
        risk_score=("risk_score", "mean"),
        peak_risk=("risk_score", "max"),
        events=("risk_score", "size"),
        first_seen=("event_time", "min"),
        last_seen=("event_time", "max"),
    )
    sessions["risk_percent"] = (sessions["risk_score"] * 100).round(1)
    sessions["peak_percent"] = (sessions["peak_risk"] * 100).round(1)
    return sessions.sort_values("risk_score", ascending=False)


@st.cache_resource
def load_model_bundle(model_path: str) -> dict:
    return joblib.load(model_path)


@st.cache_data
def load_feature_defaults(data_path: str, features: tuple) -> dict:
    reference = pd.read_csv(data_path, usecols=list(features))
    defaults = {}
    for feature in features:
        if feature == "new_sequence":
            defaults[feature] = bool(reference[feature].mode(dropna=True).iloc[0])
        else:
            defaults[feature] = float(pd.to_numeric(reference[feature], errors="coerce").median())
    return defaults


st.title("Cheat Risk Predictor")
st.caption("Enter gameplay statistics to estimate the model's risk score for one gameplay event.")

try:
    sessions = load_scored_sessions(str(DATA_PATH), str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load gameplay data or model: {exc}")
    st.stop()

high_risk_count = int((sessions["risk_score"] >= 0.7).sum())

m1, m2 = st.columns(2)
m1.metric("Sessions", f"{len(sessions):,}")
m2.metric("High risk (70%+)", f"{high_risk_count:,}")

st.subheader("Try a prediction")
st.caption("Enter one gameplay event. The model returns an event-level risk score; it does not make a moderation decision.")

try:
    model_bundle = load_model_bundle(str(MODEL_PATH))
    model_features = tuple(model_bundle["features"])
    feature_defaults = load_feature_defaults(str(DATA_PATH), model_features)
except Exception as exc:
    st.error(f"Could not prepare the prediction form: {exc}")
    st.stop()

feature_labels = {
    "yaw": "Yaw",
    "pitch": "Pitch",
    "delta_yaw": "Yaw change",
    "delta_pitch": "Pitch change",
    "accel_yaw": "Yaw acceleration",
    "accel_pitch": "Pitch acceleration",
    "target_x": "Target X",
    "target_y": "Target Y",
    "target_z": "Target Z",
    "position_x": "Player position X",
    "position_y": "Player position Y",
    "position_z": "Player position Z",
    "sensitivity": "Sensitivity",
    "time": "Event time (Unix milliseconds)",
    "new_sequence": "Starts a new sequence",
}


@st.dialog("Prediction result")
def show_prediction_result(risk_score: float) -> None:
    risk_percent = risk_score * 100
    st.metric("Estimated cheat risk", f"{risk_percent:.1f}%")
    st.progress(risk_score)
    if risk_score >= 0.85:
        st.error("High model risk — prioritize this event for human review.")
    elif risk_score >= 0.70:
        st.warning("Elevated model risk — review the gameplay evidence.")
    elif risk_score >= 0.50:
        st.info("Moderate model risk — consider reviewing with other evidence.")
    else:
        st.success("Low model risk — the model found fewer suspicious patterns in these inputs.")
    st.caption("This is a model estimate for one event, not proof of cheating or a moderation decision.")


with st.form("manual_prediction_form"):
    st.caption("Fields start with typical dataset values. Replace them with the gameplay values you want to check.")
    input_columns = st.columns(3)
    input_values = {}
    for index, feature in enumerate(model_features):
        label = feature_labels.get(feature, feature.replace("_", " ").title())
        with input_columns[index % len(input_columns)]:
            if feature == "new_sequence":
                input_values[feature] = st.checkbox(label, value=feature_defaults[feature])
            elif feature == "time":
                input_values[feature] = st.number_input(
                    label,
                    value=int(feature_defaults[feature]),
                    step=1000,
                    format="%d",
                    help="Enter the event timestamp in milliseconds, matching the dataset format.",
                )
            else:
                input_values[feature] = st.number_input(
                    label,
                    value=feature_defaults[feature],
                    step=0.1,
                    format="%.4f",
                )
    predict_clicked = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

if predict_clicked:
    prediction_row = pd.DataFrame([[input_values[name] for name in model_features]], columns=list(model_features))
    risk_score = float(model_bundle["model"].predict_proba(prediction_row)[:, 1][0])
    show_prediction_result(risk_score)
