import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('models/model.h5')

with open('models/label_encoder.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('models/one_hot_encodder_geo.pkl','rb') as file:
    one_hot_encoder_geo=pickle.load(file)

with open('models/scaler.pkl','rb') as file:
    scaler=pickle.load(file)


st.title('Customer Churn Prediction')

geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input("Balance")
credit_score=st.number_input('Credit score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('IS Active Member ',[0,1])



input_data=pd.DataFrame(
    {
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary],
    }
)
geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)


input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
predict_proba=prediction[0][0]

st.write(f"Churn Probability :{predict_proba}")
if predict_proba>0.5:
    st.write("The Customer is Likely to Churn")
else:
    st.write('The Customer is not Likely to Churn')
