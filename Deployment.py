import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.title("BANKRUPTCY PREVENTION")
st.sidebar.header("USER INPUTS")
def fun():
    industrial_risk=st.sidebar.number_input("Industrial_risk",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
    management_risk=st.sidebar.number_input("Management_risk",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
    financial_flexibility=st.sidebar.number_input("Financial_Flexibility",min_value=0.0,max_value=1.0,value=0.0,step=0.01)	
    credibility=st.sidebar.number_input("Credibility",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
    competitiveness=st.sidebar.number_input("Competitiveness",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
    operating_risk=st.sidebar.number_input("Operating_riskV",min_value=0.0,max_value=1.0,value=0.0,step=0.01)
    features={"industrial_risk":industrial_risk,"management_risk":management_risk,"financial_flexibility":financial_flexibility,
                  "credibility":credibility,"competitiveness":competitiveness,"operating_risk":operating_risk}
    dat=pd.DataFrame(features,index=[0])
    return dat
data=fun()
st.write(data)

df=pd.read_excel('Bankruptcy (2).xlsx')

lb=LabelEncoder()
df['class']=lb.fit_transform(df["class"])
x=df.drop("class",axis=1)
y=df["class"]


model=LogisticRegression()
model.fit(x,y)
pred=model.predict(data)
pre_prob=model.predict_proba(data)

st.subheader("Prediction")
st.write("Non-Bankrupt" if pre_prob[0][1]>0.5 else "Bankrupt")

st.subheader("Prediction Probability")
st.write(pre_prob)