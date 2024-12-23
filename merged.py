import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Title of the app
st.title("Predictive Maintenance Model")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

def user_input_features():
    air_temp = st.sidebar.slider("Air Temperature (°C)", 0.0, 100.0, 25.0)
    process_temp = st.sidebar.slider("Process Temperature (°C)", 0.0, 200.0, 60.0)
    rotational_speed = st.sidebar.slider("Rotational Speed (RPM)", 1000, 3000, 1500)
    torque = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 30.0)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 500, 100)
    features = {
        "Air temperature [K]": air_temp + 273.15,
        "Process temperature [K]": process_temp + 273.15,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    return pd.DataFrame(features, index=[0])

# User inputs
input_df = user_input_features()

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.sidebar.write("Dataset uploaded successfully!")

        # Ensure required columns exist
        required_columns = [
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]',
            'Machine failure'
        ]
        if all(column in data.columns for column in required_columns):
            # Preprocess data
            data['Air temperature [K]'] -= 273.15  # Convert Kelvin to Celsius
            data['Process temperature [K]'] -= 273.15  # Convert Kelvin to Celsius

            # Features and Target
            X = data[required_columns[:-1]]  # All columns except the target
            y = data['Machine failure']  # Target column

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Save the model
            joblib.dump(rf_model, 'failure_prediction_model.pkl')

            st.sidebar.success("Model trained successfully!")

            # Display user inputs
            st.subheader("User Input Parameters")
            st.write(input_df)

            # Load trained model and make prediction
            model = joblib.load("failure_prediction_model.pkl")
            prediction = model.predict(input_df)
            if prediction[0] == 1:
                st.write("⚠️ **Prediction:** High failure risk detected!")
            else:
                st.write("✅ **Prediction:** No failure risk detected.")
        else:
            st.sidebar.error("The uploaded dataset must contain the required columns.")
    except Exception as e:
        st.sidebar.error(f"Error loading dataset: {e}")
else:
    st.sidebar.info("Please upload a dataset to train the model.")
