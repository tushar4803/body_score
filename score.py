import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Body Score Prediction")

# Step 1: Upload dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())
    
    # Step 2: Preprocessing (if needed, you could add options here)
    st.write("Data Preprocessing steps will go here")
    
    # Step 3: EDA (provide visualization options)
    if st.checkbox("Show Data Distribution"):
        for col in data.select_dtypes(include=['int', 'float']).columns:
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)
    
    # Step 4: Model Selection
    st.sidebar.header("Choose Model")
    model_type = st.sidebar.selectbox("Select Model", ("Random Forest", "Decision Tree", "Logistic Regression"))
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
    
    X = data.drop("satisfaction", axis=1)  # replace with actual label column name
    y = data["satisfaction"]  # replace with actual label column name
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Step 5: Train Model
    if st.button("Train Model"):
        if model_type == "Random Forest":
            model = RandomForestClassifier()
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        else:
            model = LogisticRegression(max_iter=1000)
            
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Step 6: Display Metrics
        st.write("Accuracy:", accuracy_score(y_test, predictions))
        st.write("Classification Report", classification_report(y_test, predictions))
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
