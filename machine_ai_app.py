import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# 1) LOAD DATA
# ---------------------------
df = pd.read_csv("machine_errors_2000_rows.csv")  # replace with your CSV file

# Convert date & time to datetime
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month

# ---------------------------
# 2) LABEL ENCODING
# ---------------------------
le_error = LabelEncoder()
le_location = LabelEncoder()
le_solution = LabelEncoder()

df["ErrorTypeEncoded"] = le_error.fit_transform(df["ErrorType"])
df["LocationEncoded"] = le_location.fit_transform(df["Location"])
df["SolutionEncoded"] = le_solution.fit_transform(df["Solution"])

# ---------------------------
# 3) FEATURES & TARGET
# ---------------------------
X = df[["ErrorTypeEncoded", "LocationEncoded", "Hour", "Day", "Month"]]
y = df["SolutionEncoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 4) TRAIN MODEL
# ---------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ---------------------------
# 5) STREAMLIT DASHBOARD
# ---------------------------
st.title("⚙️ Machine Error Solution Predictor")
st.write("Predict the recommended solution for a machine error based on error type and location.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📈 Model", "🔮 Predict Solution"])

# ---------------------------
# TAB 1: Dataset
# ---------------------------
with tab1:
    st.subheader("Machine Error Logs")
    st.dataframe(df[["ErrorID", "ErrorType", "Location", "Date", "Time", "Solution"]])
    st.subheader("Dataset Summary")
    st.write(df.describe())

# ---------------------------
# TAB 2: Model Overview
# ---------------------------
with tab2:
    st.subheader("Model Information")
    st.write(f"**Model:** Random Forest Classifier")
    st.write(f"**Accuracy:** `{acc*100:.2f}%`")

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": ["ErrorType", "Location", "Hour", "Day", "Month"],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))

# ---------------------------
# TAB 3: Predict Solution
# ---------------------------
with tab3:
    st.subheader("Predict Solution for a Machine Error")

    # User inputs
    error_input = st.selectbox("Select Error Type", df["ErrorType"].unique())
    location_input = st.selectbox("Select Location", df["Location"].unique())
    hour = st.slider("Hour (0-23)", 0, 23)
    day = st.slider("Day (1-31)", 1, 31)
    month = st.slider("Month (1-12)", 1, 12)

    if st.button("Predict Solution"):
        input_data = pd.DataFrame([{
            "ErrorTypeEncoded": le_error.transform([error_input])[0],
            "LocationEncoded": le_location.transform([location_input])[0],
            "Hour": hour,
            "Day": day,
            "Month": month
        }])

        pred = model.predict(input_data)
        solution_output = le_solution.inverse_transform(pred)[0]

        st.success(f"✅ Recommended Solution: {solution_output}")
