import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = pickle.load(open(r'rf_classifier.pkl',  'rb'))
scaler = pickle.load(open(r'scaler.pkl',  'rb'))

# Prediction function
def predict_heart_disease(model, scaler, inputs):
    # Encode categorical variables
    male_encoded = 1 if inputs['gender'] == "Male" else 0
    smoker_encoded = 1 if inputs['smoker'] == "Yes" else 0
    bp_meds_encoded = 1 if inputs['bp_meds'] == "Yes" else 0
    stroke_encoded = 1 if inputs['stroke'] == "Yes" else 0
    hyp_encoded = 1 if inputs['hypertension'] == "Yes" else 0
    diabetes_encoded = 1 if inputs['diabetes'] == "Yes" else 0

    # Prepare features array
    features = np.array([[male_encoded, inputs['age'], smoker_encoded, inputs['cigs_per_day'], bp_meds_encoded,
                          stroke_encoded, hyp_encoded, diabetes_encoded, inputs['cholesterol'], inputs['sys_bp'], 
                          inputs['dia_bp'], inputs['bmi'], inputs['heart_rate'], inputs['glucose']]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    prediction = model.predict(scaled_features)

    return prediction[0], scaled_features

# Feature importance explanation
def explain_prediction(model, features):
    # Get feature importance from the model
    feature_importances = model.feature_importances_
    feature_names = ['Gender (Male)', 'Age', 'Current Smoker', 'Cigarettes per Day', 'BP Medications',
                     'Prevalent Stroke', 'Prevalent Hypertension', 'Diabetes', 'Total Cholesterol',
                     'Systolic BP', 'Diastolic BP', 'BMI', 'Heart Rate', 'Glucose']

    # Combine feature names with their importance
    explanation = {feature_names[i]: {"Value": features[0][i], "Importance": feature_importances[i]} 
                   for i in range(len(feature_names))}

    return explanation, feature_importances, feature_names

# Function to visualize input data
def visualize_input_data(inputs):
    # Create a DataFrame from inputs for visualization
    input_data = {
        'Feature': ['Age', 'Cigarettes per Day', 'Total Cholesterol', 'Systolic BP', 
                    'Diastolic BP', 'BMI', 'Heart Rate', 'Glucose'],
        'Value': [inputs['age'], inputs['cigs_per_day'], inputs['cholesterol'], 
                  inputs['sys_bp'], inputs['dia_bp'], inputs['bmi'], 
                  inputs['heart_rate'], inputs['glucose']],
        'Baseline': [40, 0, 200, 120, 80, 25, 70, 90]  # Example baseline values
    }

    # Convert to a DataFrame
    df = pd.DataFrame(input_data)

    # Create a bar plot for input values and baseline
    fig, ax = plt.subplots()
    df.set_index('Feature').plot(kind='bar', ax=ax)
    
    ax.set_title("Input Feature Values")
    ax.set_ylabel("Value")
    ax.set_xlabel("Features")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit app layout
def main():
    st.title("Heart Disease Prediction Tool")

    # Add logo
    st.image('hreat.webp', use_column_width=True)

    st.markdown("""This tool predicts the risk of heart disease based on health factors.
        Fill in the details below and click 'Predict' to get the result.""")

    # User inputs
    st.header("Patient Information")
    col1, col2 = st.columns(2)  # Create two columns for layout

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        smoker = st.selectbox("Current Smoker", ["Yes", "No"])
        cigs_per_day = st.number_input("Cigarettes per Day", min_value=0.0, value=0.0)
        bp_meds = st.selectbox("On BP Medication", ["Yes", "No"])
        heart_rate = st.number_input("Heart Rate", min_value=0.0, value=70.0)
        glucose = st.number_input("Glucose", min_value=0.0, value=80.0)

    with col2:
        stroke = st.selectbox("Prevalent Stroke", ["Yes", "No"])
        hypertension = st.selectbox("Prevalent Hypertension", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        cholesterol = st.number_input("Total Cholesterol", min_value=0.0, value=200.0)
        sys_bp = st.number_input("Systolic BP", min_value=0.0, value=120.0)
        dia_bp = st.number_input("Diastolic BP", min_value=0.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0)

    # Collect all inputs
    inputs = {
        'gender': gender, 'age': age, 'smoker': smoker, 'cigs_per_day': cigs_per_day, 'bp_meds': bp_meds,
        'stroke': stroke, 'hypertension': hypertension, 'diabetes': diabetes, 'cholesterol': cholesterol,
        'sys_bp': sys_bp, 'dia_bp': dia_bp, 'bmi': bmi, 'heart_rate': heart_rate, 'glucose': glucose
    }

    # Prediction button
    if st.button("Predict"):
        st.header("Prediction Result")
        prediction, scaled_features = predict_heart_disease(model, scaler, inputs)
        
        if prediction == 1:
            st.error("⚠️ The patient is at risk of heart disease.")
        else:
            st.success("✅ The patient is not at risk of heart disease.")

        # Explain the prediction
        st.subheader("Factors Contributing to Prediction")
        explanation, importances, feature_names = explain_prediction(model, scaled_features)
        
        # Visualize feature importance
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title("Feature Importance for Heart Disease Prediction")
        st.pyplot(fig)

        # Show detailed feature importance
        st.subheader("Detailed Breakdown:")
        for feature, details in explanation.items():
            st.write(f"**{feature}:** Value = {details['Value']}, Importance = {details['Importance']:.4f}")

        # Visualize input data
        st.subheader("Input Feature Analysis")
        visualize_input_data(inputs)

        # Health advice for risk-positive patients
        st.header("Health Advice")
        if prediction == 1:
            st.markdown("""- Consult a healthcare provider for further evaluation.
            - Consider lifestyle changes such as quitting smoking, maintaining a healthy weight, and managing blood pressure.""")
        else:
            st.markdown("""- Continue maintaining a healthy lifestyle.
            - Regular check-ups are recommended to monitor your health.""")

if __name__ == '__main__':
    main()
