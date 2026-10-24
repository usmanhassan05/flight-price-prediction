"""
✈️ Flight Price Prediction – Streamlit web app
Course: AIC354 Machine Learning Fundamentals – Assignment 3

Loads models/best_model.pkl (created by notebooks/notebook2_full_dataset_10fold.ipynb).
Every number on the "Model Comparison" tab is read from that file – nothing is hard-coded.
"""
import datetime as dt
import os

import joblib
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Data_Train.xlsx")

STUDENT = "YOUR NAME  |  YOUR REG NO"      # <- edit before submitting

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")


# ───────────────────────────── loading ─────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_excel(DATA_PATH).dropna().drop_duplicates().reset_index(drop=True)
    df["Journey_Date"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Month"] = df["Journey_Date"].dt.month
    h = df["Duration"].str.extract(r"(\d+)\s*h")[0].fillna(0).astype(int)
    m = df["Duration"].str.extract(r"(\d+)\s*m")[0].fillna(0).astype(int)
    df["Duration_Mins"] = h * 60 + m
    df["Stops"] = df["Total_Stops"].map({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4})
    return df


if not os.path.exists(MODEL_PATH):
    st.error("models/best_model.pkl not found. Run notebook 2 and put the file in the models/ folder.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
cats = bundle["categories"]
dest_by_source = bundle["destinations_by_source"]
cv = pd.DataFrame(bundle["cv_results"])
best_row = cv.iloc[0]
rmse = float(best_row["RMSE Mean"])
data = load_data()


def make_row(airline, source, dest, date, dep_time, dur_mins, stops, info):
    """Build ONE row with exactly the same features the notebook created."""
    dep_total = dep_time.hour * 60 + dep_time.minute
    arr_total = (dep_total + dur_mins) % 1440          # arrival = departure + duration
    row = {
        "Airline": airline, "Source": source, "Destination": dest, "Additional_Info": info,
        "Total_Stops": stops, "Journey_Day": date.day, "Journey_Month": date.month,
        "Journey_Weekday": date.weekday(), "Dep_Hour": dep_time.hour, "Dep_Min": dep_time.minute,
        "Arr_Hour": arr_total // 60, "Arr_Min": arr_total % 60, "Duration_Mins": dur_mins,
    }
    return pd.DataFrame([row])[FEATURES], arr_total


def fmt_inr(x):
    return f"₹{x:,.0f}"


# ───────────────────────────── header ─────────────────────────────
st.title("✈️ Flight Price Predictor")
st.caption(f"AIC354 – Machine Learning Fundamentals · Assignment 3 · {STUDENT}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", bundle["model_name"])
c2.metric(f"{bundle['k_folds']}-fold CV RMSE", fmt_inr(rmse))
c3.metric("CV R²", f"{best_row['R² Mean']:.3f}")
c4.metric("Training flights", f"{bundle['train_rows']:,}")

saved_sk = bundle.get("versions", {}).get("scikit-learn")
if saved_sk and saved_sk != sklearn.__version__:
    st.warning(f"Model was trained with scikit-learn {saved_sk}, this app runs {sklearn.__version__}. "
               f"Pin scikit-learn=={saved_sk} in requirements.txt if predictions look wrong.")

tab_pred, tab_explore, tab_models = st.tabs(["🔮 Predict", "📊 Dataset Explorer", "🏆 Model Comparison"])

# ───────────────────────────── predict ─────────────────────────────
with tab_pred:
    st.subheader("Enter flight details")
    left, right = st.columns(2)
    with left:
        airline = st.selectbox("Airline", cats["Airline"], index=cats["Airline"].index("IndiGo")
                               if "IndiGo" in cats["Airline"] else 0)
        source = st.selectbox("From", sorted(dest_by_source))
        dest = st.selectbox("To", dest_by_source[source])
        date = st.date_input("Journey date", value=dt.date(2019, 5, 18),
                             min_value=dt.date(2019, 3, 1), max_value=dt.date(2019, 6, 30),
                             help="The model was trained on flights from March–June 2019.")
    with right:
        dep_time = st.time_input("Departure time", value=dt.time(9, 0), step=300)
        d1, d2 = st.columns(2)
        hours = d1.number_input("Duration – hours", 0, 47, 2)
        mins = d2.number_input("Duration – minutes", 0, 59, 30, step=5)
        stops = st.selectbox("Stops", [0, 1, 2, 3, 4], format_func=lambda s: "Non-stop" if s == 0 else f"{s} stop(s)")
        info = st.selectbox("Additional info", cats["Additional_Info"],
                            index=cats["Additional_Info"].index("no info")
                            if "no info" in cats["Additional_Info"] else 0)

    dur_mins = int(hours) * 60 + int(mins)
    if dur_mins == 0:
        st.info("Set a duration greater than 0 to get a prediction.")
    else:
        X_new, arr_total = make_row(airline, source, dest, date, dep_time, dur_mins, stops, info)
        st.caption(f"Arrival time (departure + duration): {arr_total // 60:02d}:{arr_total % 60:02d}")

        if st.button("Predict price", type="primary"):
            price = float(model.predict(X_new)[0])
            lo, hi = max(price - rmse, 0), price + rmse
            st.success(f"### Estimated fare: {fmt_inr(price)}")
            st.write(f"Typical error of this model is about ±{fmt_inr(rmse)} (cross-validated RMSE), "
                     f"so the fare is likely between **{fmt_inr(lo)}** and **{fmt_inr(hi)}**.")

            if data is not None:
                same = data[(data["Airline"] == airline) & (data["Source"] == source) & (data["Destination"] == dest)]
                if len(same):
                    st.write(f"For comparison, the dataset's average for {airline} on {source} → {dest} "
                             f"is {fmt_inr(same['Price'].mean())} ({len(same)} flights).")
                else:
                    st.write(f"The dataset has no {airline} flights on {source} → {dest}, so this prediction is an extrapolation.")

            # same journey, every airline
            rows = [make_row(a, source, dest, date, dep_time, dur_mins, stops, info)[0] for a in cats["Airline"]]
            comp = pd.DataFrame({"Airline": cats["Airline"], "Predicted price": model.predict(pd.concat(rows))})
            comp = comp.sort_values("Predicted price")
            fig = px.bar(comp, x="Predicted price", y="Airline", orientation="h",
                         title="Same journey on every airline in the dataset (some airlines may not fly this route)")
            st.plotly_chart(fig, width="stretch")

# ───────────────────────────── explorer ─────────────────────────────
with tab_explore:
    if data is None:
        st.info("dataset/Data_Train.xlsx not found – explorer is unavailable.")
    else:
        st.subheader(f"{len(data):,} flights (after removing missing values and duplicates)")
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(data, x="Price", nbins=50, title="Price distribution"), width="stretch")
        avg_air = data.groupby("Airline")["Price"].mean().sort_values().reset_index()
        b.plotly_chart(px.bar(avg_air, x="Price", y="Airline", orientation="h", title="Average price by airline"),
                       width="stretch")
        c, d = st.columns(2)
        avg_stops = data.groupby("Stops")["Price"].mean().reset_index()
        c.plotly_chart(px.bar(avg_stops, x="Stops", y="Price", title="Average price by number of stops"),
                       width="stretch")
        d.plotly_chart(px.scatter(data, x="Duration_Mins", y="Price", color="Stops", opacity=0.5,
                                  title="Duration vs price"), width="stretch")
        with st.expander("Show raw data"):
            st.dataframe(data.drop(columns=["Journey_Date"]).head(200))

# ───────────────────────────── comparison ─────────────────────────────
with tab_models:
    st.subheader(f"{bundle['k_folds']}-fold cross-validation results (from the trained notebook)")
    show = cv.copy()
    st.dataframe(show.style.format({"RMSE Mean": "{:,.2f}", "RMSE Std": "{:,.2f}", "MAE Mean": "{:,.2f}",
                                    "R² Mean": "{:.4f}", "Time (s)": "{:.1f}"}), width="stretch")
    p, q = st.columns(2)
    p.plotly_chart(px.bar(cv, x="RMSE Mean", y="Model", error_x="RMSE Std", orientation="h",
                          title="RMSE (lower is better)"), width="stretch")
    q.plotly_chart(px.bar(cv, x="R² Mean", y="Model", orientation="h", title="R² (higher is better)"),
                   width="stretch")
    st.caption("Algorithms: Ridge, Lasso, SVR (RBF), Extra Trees, XGBoost – none used in the Car Price system.")
