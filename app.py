import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- Load Model ---------------- #
model = tf.keras.models.load_model("Models/model.h5")

with open("Models/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("Models/onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("Models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------- Sidebar ---------------- #
st.sidebar.title("🏦 Customer Churn Predictor")
st.sidebar.info(
    """
Predict whether a bank customer is likely to churn using a Deep Learning model.

Model: TensorFlow Neural Network
"""
)

# ---------------- Header ---------------- #
st.title("📊 Customer Churn Prediction")
st.caption("Fill in the customer details below and click **Predict**.")

st.divider()

# ---------------- Input Layout ---------------- #
left, right = st.columns(2)

with left:
    st.subheader("👤 Customer Details")

    geography = st.selectbox(
        "Geography",
        onehot_encoder_geo.categories_[0]
    )

    gender = st.selectbox(
        "Gender",
        label_encoder_gender.classes_
    )

    age = st.slider(
        "Age",
        18,
        92,
        35
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

with right:
    st.subheader("🏦 Account Details")

    Estimated_Salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    num_of_products = st.slider(
        "Number of Products",
        1,
        4,
        2
    )

    has_cr_card = st.radio(
        "Has Credit Card?",
        ["Yes", "No"],
        horizontal=True
    )

    is_active_member = st.radio(
        "Active Member?",
        ["Yes", "No"],
        horizontal=True
    )

# Convert radio values
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

st.divider()

# ---------------- Prediction ---------------- #
if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [Estimated_Salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    input_scaled = scaler.transform(input_data)

    probability = model.predict(input_scaled)[0][0]

    st.subheader("📈 Prediction Result")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.metric(
            "Churn Probability",
            f"{probability*100:.1f}%"
        )

    with c2:
        st.progress(float(probability))

    st.divider()

    if probability >= 0.5:
        st.error("⚠️ High Risk: Customer is likely to churn.")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    with st.expander("📋 View Processed Input"):
        st.dataframe(input_data, use_container_width=True)