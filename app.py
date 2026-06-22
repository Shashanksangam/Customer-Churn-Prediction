import streamlit as st
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np




# ==========================
# Load model and feature names
# ==========================



# ==========================
# Model Selection
# ==========================
model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]
)

if model_choice == "Logistic Regression":
    model = joblib.load("model/logistic_model.pkl")

elif model_choice == "Decision Tree":
    model = joblib.load("model/decision_tree_model.pkl")

else:
    model = joblib.load("model/random_forest_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

logistic_model = joblib.load("model/logistic_model.pkl")
decision_model = joblib.load("model/decision_tree_model.pkl")
random_model = joblib.load("model/random_forest_model.pkl")

lr_accuracy = 80.0
dt_accuracy = 78.0
rf_accuracy = 82.0

# ==========================
# Sidebar
# ==========================
st.sidebar.title("About Project")

st.sidebar.info("""
Customer Churn Prediction using Machine Learning

Models Used:
• Logistic Regression
• Decision Tree
• Random Forest

Dataset Size:
7010 Customers

Developer:
Shashank Sangam
B.Sc Computer Science & Mathematics
""")


st.sidebar.markdown("---")

st.sidebar.markdown("""
### Developer

**Shashank Sangam**

B.Sc Computer Science & Mathematics

Dayalbagh Educational Institute

Future Data Scientist 🚀
""")


st.sidebar.markdown("---")

st.sidebar.markdown("""
### Links

GitHub:
https://github.com/Shashanksangam

LinkedIn:
https://linkedin.com/in/shashank-sangam-72b3853a0
""")




# ==========================
# Main Title
# ==========================
st.markdown(
"""
<h1 style='text-align:center;color:#00C0FF'>
📊 Customer Churn Prediction System
</h1>
""",
unsafe_allow_html=True
)

st.caption(
"Predicting customer retention using multiple machine learning models and interactive analytics."
)

# ==========================
# Numerical Inputs
# ==========================
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)
total_charges = st.number_input("Total Charges", value=500.0)
senior_citizen = st.selectbox("Senior Citizen", [0, 1])

# ==========================
# Categorical Inputs
# ==========================
gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
phone_service = st.selectbox("Phone Service", ["No", "Yes"])

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

# ==========================
# Prediction Button
# ==========================
if st.button("Predict"):

    # Create dataframe with all columns
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)

    # Numerical features
    input_data["SeniorCitizen"] = senior_citizen
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Gender
    if gender == "Male":
        input_data["gender_Male"] = 1

    # Partner
    if partner == "Yes":
        input_data["Partner_Yes"] = 1

    # Dependents
    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    # Phone Service
    if phone_service == "Yes":
        input_data["PhoneService_Yes"] = 1

    # Multiple Lines
    if multiple_lines == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif multiple_lines == "No phone service":
        input_data["MultipleLines_No phone service"] = 1

    # Internet Service
    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    # Online Security
    if online_security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    # Online Backup
    if online_backup == "Yes":
        input_data["OnlineBackup_Yes"] = 1
    elif online_backup == "No internet service":
        input_data["OnlineBackup_No internet service"] = 1

    # Device Protection
    if device_protection == "Yes":
        input_data["DeviceProtection_Yes"] = 1
    elif device_protection == "No internet service":
        input_data["DeviceProtection_No internet service"] = 1

    # Tech Support
    if tech_support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    # Streaming TV
    if streaming_tv == "Yes":
        input_data["StreamingTV_Yes"] = 1
    elif streaming_tv == "No internet service":
        input_data["StreamingTV_No internet service"] = 1

    # Streaming Movies
    if streaming_movies == "Yes":
        input_data["StreamingMovies_Yes"] = 1
    elif streaming_movies == "No internet service":
        input_data["StreamingMovies_No internet service"] = 1

    # Contract
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    # Paperless Billing
    if paperless_billing == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    # Payment Method
    if payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1

    # ==========================
    # Prediction
    # ==========================
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    stay_probability = probability[0][0] * 100
    churn_probability = probability[0][1] * 100

    st.divider()
    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:

        st.error("⚠ Customer is likely to Churn")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2f}%"
        )

        st.progress(int(churn_probability))

        if churn_probability >= 80:
            st.error("🔴 HIGH RISK CUSTOMER")

        elif churn_probability >= 50:
            st.warning("🟠 MEDIUM RISK CUSTOMER")

        else:
            st.info("🟡 LOW RISK CUSTOMER")

    else:

        st.success("✅ Customer is likely to Stay")

        st.metric(
            "Stay Probability",
            f"{stay_probability:.2f}%"
        )

        st.progress(int(stay_probability))

        st.success("🟢 CUSTOMER RETENTION IS STRONG")


     # ==========================
     # Phase 17 : Pie Chart
     # ==========================
        st.divider()
        
        st.subheader("📊 Churn Probability Visualization")
        
        labels = ["Stay", "Churn"]
        sizes = [stay_probability, churn_probability]
        
        fig, ax = plt.subplots()
        
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%"
        )
        
        ax.axis("equal")
        
        st.pyplot(fig)


    # ==========================
    # Phase 18 : Feature Importance
    # ==========================
    st.divider()
    
    st.subheader("📈 Feature Importance")
    
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.coef_[0]
    })
    
    importance["Importance"] = importance["Importance"].abs()
    
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)
    
    fig2, ax2 = plt.subplots(figsize=(8,5))
    
    ax2.barh(
        importance["Feature"],
        importance["Importance"]
    )
    
    ax2.invert_yaxis()
    
    st.pyplot(fig2)

    # ==========================
    # Customer Summary
    # ==========================
    st.divider()

    st.subheader("📋 Customer Information Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Personal Information")
        st.write(f"Gender: {gender}")
        st.write(f"Senior Citizen: {senior_citizen}")
        st.write(f"Partner: {partner}")
        st.write(f"Dependents: {dependents}")

    with col2:
        st.write("### Charges")
        st.write(f"Tenure: {tenure} months")
        st.write(f"Monthly Charges: ${monthly_charges}")
        st.write(f"Total Charges: ${total_charges}")

    st.divider()

    st.write("### Services")

    st.write(f"Phone Service: {phone_service}")
    st.write(f"Multiple Lines: {multiple_lines}")
    st.write(f"Internet Service: {internet_service}")
    st.write(f"Online Security: {online_security}")
    st.write(f"Online Backup: {online_backup}")
    st.write(f"Device Protection: {device_protection}")
    st.write(f"Tech Support: {tech_support}")
    st.write(f"Streaming TV: {streaming_tv}")
    st.write(f"Streaming Movies: {streaming_movies}")

    st.divider()

    st.write("### Billing")

    st.write(f"Contract: {contract}")
    st.write(f"Paperless Billing: {paperless_billing}")
    st.write(f"Payment Method: {payment_method}")

    st.divider()

    

    # ==========================
    # Download Report
    # ==========================
    report = f"""
Customer Churn Prediction Report

Gender: {gender}
Partner: {partner}
Dependents: {dependents}
Senior Citizen: {senior_citizen}

Tenure: {tenure}
Monthly Charges: {monthly_charges}
Total Charges: {total_charges}

Phone Service: {phone_service}
Internet Service: {internet_service}
Contract: {contract}
Payment Method: {payment_method}

Prediction: {"Churn" if prediction[0] == 1 else "Stay"}

Churn Probability: {churn_probability:.2f}%
Stay Probability: {stay_probability:.2f}%

Developer:
Shashank Sangam
"""

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="customer_report.txt",
        mime="text/plain"
    )


    # ==========================
    # Phase 19 : Model Comparison
    # ==========================
    st.divider()
    
    st.subheader("🤖 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Logistic Regression",
            f"{lr_accuracy}%"
        )
    
    with col2:
        st.metric(
            "Decision Tree",
            f"{dt_accuracy}%"
        )
    
    with col3:
        st.metric(
            "Random Forest",
            f"{rf_accuracy}%"
        )


        # ==========================
    # Phase 20 : Save Prediction History
    # ==========================
    
    history_row = pd.DataFrame({
    
        "Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    
        "Gender": [gender],
    
        "Tenure": [tenure],
    
        "MonthlyCharges": [monthly_charges],
    
        "TotalCharges": [total_charges],
    
        "Prediction": [
            "Churn" if prediction[0] == 1 else "Stay"
        ],
    
        "Probability (%)": [
            round(churn_probability, 2)
        ]
    })
    
    if os.path.exists("prediction_history.csv"):
    
        old_data = pd.read_csv("prediction_history.csv")
    
        history_data = pd.concat(
            [old_data, history_row],
            ignore_index=True
        )
    
    else:
    
        history_data = history_row
    
    history_data.to_csv(
        "prediction_history.csv",
        index=False
    )


    st.divider()
    
    st.subheader("📜 Prediction History")
    
    history = pd.read_csv("prediction_history.csv")
    
    st.dataframe(history)


   
    
    # ==========================
    # Phase 21 : Search History
    # ==========================
    
    st.divider()
    
    st.subheader("🔍 Search Prediction History")
    
    search_text = st.text_input(
        "Search customer history"
    )
    
    if search_text:
    
        filtered_history = history[
            history.astype(str)
            .apply(
                lambda x: x.str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            )
            .any(axis=1)
        ]
    
        st.dataframe(filtered_history)
    
    else:
    
        st.dataframe(history)



    # ==========================
    # Phase 22 : Analytics Dashboard
    # ==========================
    
    st.divider()
    st.subheader("📊 Analytics Dashboard")
    
    # Prediction counts
    prediction_counts = history["Prediction"].value_counts()
    
    st.write("### Churn vs Stay Count")
    st.bar_chart(prediction_counts)
    
    # Probability distribution
    if "Probability (%)" in history.columns:
    
        st.write("### Probability Distribution")
    
        probability_chart = history[["Probability (%)"]]
    
        st.line_chart(probability_chart)
    
    # Tenure distribution
    if "Tenure" in history.columns:
    
        st.write("### Tenure Distribution")
    
        tenure_chart = history[["Tenure"]]
    
        st.area_chart(tenure_chart)


        st.caption(
        "Developed by Shashank Sangam | B.Sc Computer Science & Mathematics"
    )


        st.divider()




    # ==========================
    # Feature Importance
    # ==========================
    st.divider()
    st.subheader("📈 Feature Importance")
    
    try:
    
        # Logistic Regression
        if hasattr(model, "coef_"):
    
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": abs(model.coef_[0])
            })
    
        # Decision Tree / Random Forest
        elif hasattr(model, "feature_importances_"):
    
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })
    
        importance_df = (
            importance_df
            .sort_values("Importance", ascending=False)
            .head(10)
        )
    
        st.dataframe(importance_df)
    
        st.bar_chart(
            importance_df.set_index("Feature")
        )
    
    except Exception as e:
    
        st.warning(
            f"Feature importance unavailable.\n\n{e}"
        )



        
    
    st.markdown(
    """
    <center>
    
    ### Developed by Shashank Sangam
    
    Customer Churn Prediction using Machine Learning
    
    Logistic Regression • Decision Tree • Random Forest
    
    2026
    
    </center>
    """,
    unsafe_allow_html=True
    )