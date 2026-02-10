import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_dataset(url):
    """Load dataset from a given URL."""
    return pd.read_csv(url)

def main():
    st.title("EDA and ML App")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["EDA", "ML Modeling"])

    if page == "EDA":
        eda_page()
    elif page == "ML Modeling":
        ml_modeling_page()

def eda_page():
    st.header("Exploratory Data Analysis")

    dataset_url = st.selectbox("Select Dataset", [
        "https://raw.githubusercontent.com/josephgitau/project_defcone/refs/heads/main/Data/Insurance/cleaned_group_death_claims.csv",
        "https://raw.githubusercontent.com/josephgitau/project_defcone/refs/heads/main/Data/Insurance/cleaned_individual_death_claims.csv"
    ])

    st.write("Loading dataset...")
    data = load_dataset(dataset_url)

    st.write("Dataset Preview:")
    st.dataframe(data.head())
    st.write("Summary Statistics:")
    st.write(data.describe())
    st.write("Column Information:")
    st.write(data.info())

def ml_modeling_page():
    st.header("Machine Learning Modeling")

    dataset_url = st.selectbox("Select Dataset", [
        "https://raw.githubusercontent.com/josephgitau/project_defcone/refs/heads/main/Data/Insurance/cleaned_group_death_claims.csv",
        "https://raw.githubusercontent.com/josephgitau/project_defcone/refs/heads/main/Data/Insurance/cleaned_individual_death_claims.csv"
    ])

    st.write("Loading dataset...")
    data = load_dataset(dataset_url)

    st.write("Dataset Preview:")
    st.dataframe(data.head())

    features = st.multiselect("Features", data.columns.tolist())
    target = st.selectbox("Target", data.columns.tolist())

    if features and target:
        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
