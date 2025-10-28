import streamlit as st
import requests
import json

# IMPORTANT: Paste your Vercel API URL here
API_URL = "https://wine-quality-ml-project.vercel.app/predict"

st.set_page_config(layout="wide")
st.title("🍷 Red Wine Quality Predictor")
st.write("Input the chemical properties of a red wine to predict its quality.")

st.sidebar.header("Wine Features")

def get_user_inputs():
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.9, 16.0, 1.9, 0.1)

    with col2:
        chlorides = st.slider("Chlorides", 0.01, 0.62, 0.076, 0.001)
        free_sulfur = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 11.0, 1.0)
        total_sulfur = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 34.0, 1.0)
        density = st.slider("Density", 0.990, 1.004, 0.9978, 0.0001, format="%.4f")

    with col3:
        ph = st.slider("pH", 2.7, 4.0, 3.51, 0.01)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56, 0.01)
        alcohol = st.slider("Alcohol", 8.4, 15.0, 9.4, 0.1)

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur,
        "total sulfur dioxide": total_sulfur,
        "density": density,
        "pH": ph,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    return data

input_data = get_user_inputs()

st.subheader("Prediction")
if st.button("Predict Wine Quality"):
    if not API_URL.startswith("https"):
        st.error("Please update the API_URL in the streamlit_app.py file with your Vercel deployment URL.")
    else:
        try:
            payload = json.dumps(input_data)
            headers = {"Content-Type": "application/json"}

            response = requests.post(API_URL, data=payload, headers=headers)

            if response.status_code == 200:
                prediction = response.json()['prediction']
                if prediction == "Good Quality":
                    st.success(f"Prediction: **{prediction}** 🎉")
                else:
                    st.error(f"Prediction: **{prediction}** 👎")
            else:
                st.error(f"Error from API (Status {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to API: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")