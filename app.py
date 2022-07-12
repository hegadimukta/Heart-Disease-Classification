import streamlit as st
import pickle

#loading the models here
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
randomforest_model = pickle.load(open('randomforest_model.pkl', 'rb'))
decisiontree_model = pickle.load(open('decisiontree_model.pkl', 'rb'))
adaboost_model = pickle.load(open('adaboost_model.pkl', 'rb'))
xgboost_model = pickle.load(open('xgboost_model.pkl', 'rb'))


def classify(num):
    if num==0:
        st.success('It is very likely that the patient does not have a heart disease')
    else:
        st.error('It is very likely that the patient has a heart disease')


def main():

    st.set_page_config(page_title="Heart Disease Classification",layout="centered",initial_sidebar_state="expanded")

    st.title("Heart Disease Classification")

    classifiers = ['Logistic Regression','K Nearest Neighbours','Decision Tree','Random Forest', 'AdaBoost', 'XGBoost']

    classifier_selected = st.sidebar.selectbox('Select a classification model', classifiers)

    st.subheader(classifier_selected) 

    age = st.slider('Age (measured in years)', 25, 100, 40, 1)
    trestbps = st.slider('Resting Blood Pressure (measured in mm Hg)', 80, 200, 100, 1)
    cholestrol = st.slider('Cholestrol (measured in mg/dl)', 100, 564, 120, 1)
    thalch = st.slider("Heart rate achieved during patient's stress testing", 70, 210, 100, 1)
    oldpeak = st.slider('Stress Test Depression', 0.0, 6.2, 0.0, 0.1)
    ca = st.slider('Number of major vessels (0-3) colored by fluoroscopy', 0, 3, 0, 1)
    gender = st.slider('1=Male and 0=Female', 0, 1, 0, 1)
    cp = st.slider('Type of Chest Pain', 0, 3, 0, 1)
    restecg = st.slider('Resting Electrocardiogram', 0, 2, 0, 1)
    exang = st.slider('Exercise induced angina (0=No; 1=Yes)', 0, 1, 0, 1)
    slope = st.slider('Slope for peak exercise', 0, 2, 0, 1)
    thal = st.slider('Thallium Heart Rate', 0, 2, 0, 1)
    fbs = st.slider('Fasting blood sugar', 0, 1, 0, 1)

    inputs = [[age, trestbps, cholestrol, thalch, oldpeak, ca, gender, cp, restecg, exang, slope, thal, fbs]]

    if st.button('Classify'):
        if classifier_selected=='Logistic Regression':
            classify(logistic_model.predict(inputs))
        elif classifier_selected=='K Nearest Neighbours':
            classify(knn_model.predict(inputs))
        elif classifier_selected=='Decision Tree':
            classify(decisiontree_model.predict(inputs))
        elif classifier_selected=='Random Forest':
            classify(randomforest_model.predict(inputs))
        elif classifier_selected=='AdaBoost':
            classify(adaboost_model.predict(inputs))
        elif classifier_selected=='XGBoost':
            classify(xgboost_model.predict(inputs))


if __name__=='__main__':
    main()