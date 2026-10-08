from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "final_dataset.csv"
MODEL_PATH = ROOT / "cheat_detection_model.joblib"

st.set_page_config(page_title="Gameplay Risk Assessment", page_icon="🎮", layout="wide")


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


st.title("Gameplay Risk Assessment")
st.caption("A machine-learning assisted tool for estimating suspicious gameplay patterns from event statistics.")

try:
    sessions = load_scored_sessions(str(DATA_PATH), str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load gameplay data or model: {exc}")
    st.stop()

high_risk_count = int((sessions["risk_score"] >= 0.7).sum())
event_count = int(sessions["events"].sum())

m1, m2, m3 = st.columns(3)
m1.metric("Sessions analyzed", f"{len(sessions):,}")
m2.metric("Gameplay events", f"{event_count:,}")
m3.metric("Elevated session scores", f"{high_risk_count:,}", help="Sessions with an average model score of 70% or higher.")
st.caption("Session counts summarize the supplied dataset. A model score is an estimate, not a finding of cheating.")

st.divider()
st.subheader("Assess one gameplay event")
st.caption("Enter the event statistics below, then run the model to view its estimated risk score.")

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


@st.dialog("Gameplay risk result")
def show_prediction_result(risk_score: float) -> None:
    risk_percent = risk_score * 100
    score_column, explanation_column = st.columns([1, 2])
    score_column.metric("Estimated risk", f"{risk_percent:.1f}%")
    score_column.progress(risk_score, text="Model score")
    if risk_score >= 0.85:
        explanation_column.error("High risk score")
        explanation_column.write("The model found patterns that are uncommon in the legitimate examples. Review the gameplay evidence carefully.")
    elif risk_score >= 0.70:
        explanation_column.warning("Elevated risk score")
        explanation_column.write("Consider checking this event alongside other gameplay evidence.")
    elif risk_score >= 0.50:
        explanation_column.info("Moderate risk score")
        explanation_column.write("The model found some patterns worth considering, but this score alone is inconclusive.")
    else:
        explanation_column.success("Lower risk score")
        explanation_column.write("The model found fewer suspicious patterns in these inputs.")
    st.divider()
    st.caption("This is an event-level model estimate, not proof of cheating or a moderation decision.")


with st.form("manual_prediction_form"):
    st.caption("Fields are prefilled with typical values from the dataset. Replace them with the event you want to assess.")
    input_values = {}
    field_groups = [
        ("Aim and movement", ["yaw", "pitch", "delta_yaw", "delta_pitch", "accel_yaw", "accel_pitch"]),
        ("Target and player position", ["target_x", "target_y", "target_z", "position_x", "position_y", "position_z"]),
        ("Session details", ["sensitivity", "time", "new_sequence"]),
    ]
    grouped_features = {feature for _, features in field_groups for feature in features}
    additional_features = [feature for feature in model_features if feature not in grouped_features]
    if additional_features:
        field_groups.append(("Additional model inputs", additional_features))

    for group_name, group_features in field_groups:
        group_features = [feature for feature in group_features if feature in model_features]
        if not group_features:
            continue
        with st.container(border=True):
            st.markdown(f"#### {group_name}")
            for start in range(0, len(group_features), 2):
                input_columns = st.columns(2)
                for column, feature in zip(input_columns, group_features[start:start + 2]):
                    label = feature_labels.get(feature, feature.replace("_", " ").title())
                    with column:
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
    predict_clicked = st.form_submit_button("Run risk assessment", type="primary", use_container_width=True)

if predict_clicked:
    prediction_row = pd.DataFrame([[input_values[name] for name in model_features]], columns=list(model_features))
    risk_score = float(model_bundle["model"].predict_proba(prediction_row)[:, 1][0])
    show_prediction_result(risk_score)
