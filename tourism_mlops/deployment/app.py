import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="satyabrd123/tourism-prediction-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("🌴 Wellness Tourism Package Purchase Predictor")
st.write("The Wellness Tourism Package Predictor is an internal tool for sales teams to predict whether customers are likely to purchase tourism packages based on their details.")
st.write("Kindly enter the customer details to check their purchase likelihood.")

# Collect user input
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    NumberOfTrips = st.number_input("Number of Trips", min_value=0, value=2)
    Passport = st.selectbox("Has Passport?", [0, 1])
    PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("Owns Car?", [0, 1])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

# Encode categorical inputs (must match training encoding)
occupation_map = {"Salaried": 0, "Small Business": 1, "Large Business": 2, "Free Lancer": 3}
gender_map = {"Male": 0, "Female": 1}
product_map = {"Basic": 0, "Standard": 1, "Deluxe": 2, "Super Deluxe": 3, "King": 4}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2, "Unmarried": 3}
designation_map = {"Executive": 0, "Manager": 1, "Senior Manager": 2, "AVP": 3, "VP": 4}
contact_map = {"Self Enquiry": 0, "Company Invited": 1}

input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': occupation_map[Occupation],
    'Gender': gender_map[Gender],
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': product_map[ProductPitched],
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': marital_map[MaritalStatus],
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': designation_map[Designation],
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': contact_map[TypeofContact]
}])

# Predict button
if st.button("Predict Purchase Likelihood"):
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        result = "purchase the package" if prediction == 1 else "not purchase the package"
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        if prediction == 1:
            st.success(f"✅ Based on the information provided, the customer is likely to **{result}**.")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.write("💡 **Recommendation:** Prioritize this lead for follow-up.")
        else:
            st.warning(f"❌ Based on the information provided, the customer is likely to **{result}**.")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            st.write("💡 **Recommendation:** Consider additional engagement strategies or alternative offerings.")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
