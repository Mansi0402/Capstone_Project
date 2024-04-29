import json
import requests
import streamlit as st
import pickle as pk
import joblib
import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import missingno as msno
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
# App title
st.title("Financial Fraud Detection App using Machine Learning")

#some image
st.image("img/credit_card_fraud.jpg")

# Description
st.write(
    """
    ## About
    
    With the growth of e-commerce websites, people and financial companies rely on online services
    to carry out their transactions that have led to an exponential increase in the credit card frauds.
    Fraudulent credit card transactions lead to a loss of huge amount of money. The design of an
    effective fraud detection system is necessary in order to reduce the losses incurred by the
    customers and financial companies. 

    **This Streamlit App  utilizes a Machine Learning model(XGBoost) API to detect potential fraud in credit card transactions.**

    """
)


def type_transaction(content):
    if content == "PAYMENT":
        content = 0
    elif content == "TRANSFER":
        content = 1
    elif content == "CASH_OUT":
        content = 2
    elif content == "DEBIT":
        content = 3
    elif content == "CASH_IN":
        content = 4
    return content


st.sidebar.header("Input user and transaction information")

# User data
sender_name = st.sidebar.text_input(" Sender Name ID")
receiver_name = st.sidebar.text_input(" Receiver Name ID")

## Transaction information
type_lebels = ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")
type = st.sidebar.selectbox(" Type of transaction", type_lebels)

step = st.sidebar.slider("Number of Hours it took the Transaction to complete:", min_value = 0, max_value = 744)

amount = st.sidebar.number_input("Amount in $",min_value=0, max_value=110000)
oldbalanceorg = st.sidebar.number_input("""Sender Balance Before Transaction was made""",min_value=0, max_value=110000)
newbalanceorg = st.sidebar.number_input("""Sender Balance After Transaction was made""",min_value=0, max_value=110000)
oldbalancedest = st.sidebar.number_input("""Recipient Balance Before Transaction was made""",min_value=0, max_value=110000)
newbalancedest = st.sidebar.number_input("""Recipient Balance After Transaction was made""",min_value=0, max_value=110000)
## flag 
isflaggedfraud = "Non fraudulent"
if amount >= 200000:
  isflaggedfraud = "Fraudulent transaction"
else:
  isflaggedfraud = "Non fraudulent"


result_button = st.button("Detect Result")


if result_button:

    ## Features
    data = {
        "step": step,
        "type": type_transaction(type),
        "amount": amount,
        "oldbalanceOrg": oldbalanceorg,
        "newbalanceOrig": newbalanceorg,
        "oldbalanceDest": oldbalancedest,
        "newbalanceDest": newbalancedest
    }

    ## Transaction detail
    st.write(
        f""" 
        ## **Transaction Details**

        #### **User informantion**:

        Sender Name(ID): {sender_name}\n
        Receiver Name(ID): {receiver_name}

        #### **Transaction information**:

        Number of Hours it took to complete: {step}\n
        Type of Transaction: {type}\n
        Amount Sent: {amount}$\n
        Sender Balance Before Transaction: {oldbalanceorg}$\n
        Sender Balance After Transaction: {newbalanceorg}$\n
        Recepient Balance Before Transaction: {oldbalancedest}$\n
        Recepient Balance After Transaction: {newbalancedest}$\n
        System Flag Fraud Status(Transaction amount greater than $200000): {isflaggedfraud}

        """
    )

    st.write("""## **Prediction**""")
    
    try:
    # Load RandomForest model
        model = joblib.load('RandomForestModel.joblib')
    except FileNotFoundError:
        st.error("Error: Model file not found. Please ensure the file path is correct.")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # Check if model is loaded successfully
    if 'model' in locals():
        # Add your inference code here
        features = pd.DataFrame(data, index=[0])

        # Inference - Predictions
        pred = model.predict(features)
        pred_prob = model.predict_proba(features) 

        prob_nofraud = np.round(pred_prob[0, 0] * 100, 2)
        prob_fraud = np.round(pred_prob[0, 1] * 100, 2)

        if pred == 1:
            st.header(f"is potentially fraudulent with a probability of {prob_fraud} percent")
        else:
            st.header(f"is not potentially fraudulent with a probability of {prob_nofraud} percent")
    else:
        st.error("Failed to load the model. Please check the error message above for details.")

        



