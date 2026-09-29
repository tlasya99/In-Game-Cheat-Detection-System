from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "final_dataset.csv"
MODEL_PATH = ROOT / "cheat_detection_model.joblib"
REVIEWS_PATH = ROOT / "review_actions.csv"
ACTIONS = ["Needs investigation", "No violation", "Confirmed cheating"]

st.set_page_config(page_title="Fair Play Review", page_icon="🎮", layout="wide")


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


def load_reviews() -> pd.DataFrame:
    columns = ["uuid", "action", "note", "reviewed_at"]
    if not REVIEWS_PATH.exists():
        return pd.DataFrame(columns=columns)
    reviews = pd.read_csv(REVIEWS_PATH).reindex(columns=columns)
    return reviews.drop_duplicates("uuid", keep="last")


def save_review(uuid: str, action: str, note: str) -> None:
    reviews = load_reviews()
    reviews = reviews[reviews["uuid"].astype(str) != uuid]
    new_review = pd.DataFrame([{
        "uuid": uuid,
        "action": action,
        "note": note.strip(),
        "reviewed_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }])
    pd.concat([reviews, new_review], ignore_index=True).to_csv(REVIEWS_PATH, index=False)


st.title("Fair Play Review")
st.caption("Gameplay sessions ranked by the cheat detection model. Scores prioritize moderator review; they are not proof of cheating.")

try:
    sessions = load_scored_sessions(str(DATA_PATH), str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load gameplay data or model: {exc}")
    st.stop()

reviews = load_reviews()
if not reviews.empty:
    sessions = sessions.merge(reviews, on="uuid", how="left")
else:
    sessions["action"] = pd.NA
    sessions["note"] = pd.NA
    sessions["reviewed_at"] = pd.NA
sessions["review_status"] = sessions["action"].fillna("Unreviewed")

high_risk_count = int((sessions["risk_score"] >= 0.7).sum())
unreviewed_count = int(sessions["action"].isna().sum())
confirmed_count = int((sessions["action"] == "Confirmed cheating").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sessions", f"{len(sessions):,}")
m2.metric("High risk (70%+)", f"{high_risk_count:,}")
m3.metric("Awaiting review", f"{unreviewed_count:,}")
m4.metric("Confirmed by moderators", f"{confirmed_count:,}")

st.subheader("Session risk overview")
risk_bands = ["0–49%", "50–69%", "70–84%", "85–100%"]
sessions["risk_band"] = pd.cut(
    sessions["risk_score"],
    bins=[0, 0.5, 0.7, 0.85, 1.000001],
    labels=risk_bands,
    right=False,
    include_lowest=True,
)
risk_counts = (
    sessions["risk_band"]
    .value_counts()
    .reindex(risk_bands, fill_value=0)
    .rename_axis("Average risk range")
    .to_frame("Sessions")
)
st.bar_chart(risk_counts, y_label="Number of sessions")
st.caption("Bars count sessions by average model risk score. Risk scores are review signals, not confirmation of cheating.")

st.subheader("Review queue")
f1, f2 = st.columns([1, 1])
minimum_risk = f1.slider("Minimum risk score", min_value=0, max_value=100, value=50, step=5)
status_options = ["All", "Unreviewed"] + ACTIONS
status_filter = f2.selectbox("Review status", status_options)
queue = sessions[sessions["risk_score"] >= minimum_risk / 100]
if status_filter == "Unreviewed":
    queue = queue[queue["action"].isna()]
elif status_filter != "All":
    queue = queue[queue["action"] == status_filter]

visible = queue[["uuid", "risk_percent", "peak_percent", "events", "review_status"]].rename(columns={
    "uuid": "Session ID", "risk_percent": "Average risk %", "peak_percent": "Peak risk %",
    "events": "Events", "review_status": "Review status",
})
st.dataframe(visible, hide_index=True, use_container_width=True)

st.subheader("Review a session")
if queue.empty:
    st.info("No sessions match these filters.")
else:
    session_ids = queue["uuid"].astype(str).tolist()
    selected_uuid = st.selectbox("Session", session_ids, format_func=lambda value: f"{value} · {sessions.loc[sessions.uuid == value, 'risk_percent'].iloc[0]:.1f}% risk")
    selected = sessions.loc[sessions["uuid"] == selected_uuid].iloc[0]
    left, right = st.columns(2)
    left.metric("Average model risk", f"{selected['risk_percent']:.1f}%")
    right.metric("Highest event risk", f"{selected['peak_percent']:.1f}%")
    st.caption(f"{int(selected['events'])} gameplay events in this session")

    current_action = selected["action"] if pd.notna(selected["action"]) and selected["action"] in ACTIONS else ACTIONS[0]
    with st.form("review_form"):
        action = st.selectbox("Moderator decision", ACTIONS, index=ACTIONS.index(current_action))
        current_note = selected["note"] if pd.notna(selected["note"]) else ""
        note = st.text_area("Review notes", value=current_note, max_chars=2000)
        submitted = st.form_submit_button("Save review")
        if submitted:
            save_review(selected_uuid, action, note)
            st.success("Review saved.")
            st.rerun()

with st.expander("About this dashboard"):
    st.write("The model score is the average event level probability for a session. A moderator should review the gameplay evidence before taking action. Decisions are saved locally in review_actions.csv.")
    st.write("This prototype does not include login or role based access. Add authentication and audit controls before using it with real accounts.")
