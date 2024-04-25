import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
@st.cache  # Add caching so that data is loaded only once
def load_data():
    data = pd.read_csv('seed_data.csv')
    return data

# Function to preprocess and split the data
def preprocess_data(data):
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train the KNN model
def train_model(X_train_scaled, y_train, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    return knn

# Function to evaluate the model and display results
def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write('Confusion Matrix:')
    st.write(confusion_matrix(y_test, y_pred))
    st.write('Classification Report:')
    st.write(classification_report(y_test, y_pred))

# Function to predict wheat kernel type based on user input
def predict_kernel_type(model, scaler):
    st.header('Predict Wheat Kernel Type')
    
    # Input values from user
    st.subheader('Enter Wheat Kernel Features:')
    area = st.number_input('Area', min_value=10.0, max_value=25.0, value=15.0, format="%.4f")
    perimeter = st.number_input('Perimeter', min_value=10.0, max_value=25.0, value=15.0, format="%.4f")
    compactness = st.number_input('Compactness', min_value=0.7, max_value=1.0, value=0.85, format="%.4f")
    kernel_length = st.number_input('Kernel Length', min_value=4.0, max_value=7.0, value=5.0, format="%.4f")
    kernel_width = st.number_input('Kernel Width', min_value=2.0, max_value=5.0, value=3.0, format="%.4f")
    asymmetry_coeff = st.number_input('Asymmetry Coefficient', min_value=0.0, max_value=10.0, value=0.0, format="%.4f")
    kernel_groove = st.number_input('Kernel Groove', min_value=4.0, max_value=7.0, value=5.0, format="%.4f")
    
    # Make prediction based on user input
    input_data = np.array([[area, perimeter, compactness, kernel_length, kernel_width, asymmetry_coeff, kernel_groove]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    # Display predicted wheat kernel type
    kernel_types = ['Kama', 'Rosa', 'Canadian']
    if prediction == 0:
        st.write(f'Predicted Wheat Kernel Type: {kernel_types[0]}')
    elif prediction == 1:
        st.write(f'Predicted Wheat Kernel Type: {kernel_types[1]}')
    elif prediction == 2:
        st.write(f'Predicted Wheat Kernel Type: {kernel_types[2]}')

def main():
    st.title('Wheat Kernel Classification with KNN')

    # Load data
    data = load_data()

    # Preprocess and split data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data)

    # Train the KNN model
    k_value = 20
    knn_model = train_model(X_train_scaled, y_train, k=k_value)

    # # Evaluate the model
    # st.header('Model Evaluation')
    # evaluate_model(knn_model, X_test_scaled, y_test)

    # Predict wheat kernel type based on user input
    predict_kernel_type(knn_model, scaler)

if __name__ == '__main__':
    main()