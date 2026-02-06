import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy import stats


# PAGE CONFIG

st.set_page_config(page_title="AI Data Analyzer", layout="wide")
st.title("📊 AI Data Analyzer")
st.write("Upload CSV or Excel file → Complete EDA → Data Cleaning → ML Ready Dataset")

# FILE UPLOAD

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

if uploaded_file is not None:

    # READ DATA
    
    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, encoding= 'utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding= 'latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding= 'ISO-8859-1')
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.astype(str)
    else:
        st.error("❌ Unsupported file type")
        st.stop() 
            
    df.columns = df.columns.astype(str)
    st.success("✅ File Uploaded Successfully")

    # STEP 1: EDA
    
    st.header("🔍 STEP 1: Exploratory Data Analysis (EDA)")

    st.subheader("1️⃣ Viewing the Data")
    st.write("Head")
    st.dataframe(df.head())

    st.write("Tail")
    st.dataframe(df.tail())

    st.write(f"**Shape:** {df.shape}")
    st.write("**Info:**")
    st.text(df.info())

    st.write("**Columns & Data Types**")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    st.subheader("2️⃣ Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("3️⃣ Value Counts (Categorical)")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        st.write(f"**{col}**")
        st.dataframe(df[col].value_counts())

    st.subheader("4️⃣ Missing Value Analysis")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Values": missing,
        "Percent (%)": missing_percent
    })
    st.dataframe(missing_df)

    st.subheader("5️⃣ Visualizations")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Histograms
    st.write("📊 Histograms")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        plt.xticks(rotation= 45)
        plt.tight_layout()
        ax.set_title(col)
        st.pyplot(fig)

    # Barplots
    st.write("📊 Bar Plots")
    for col in categorical_cols:
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        plt.xticks(rotation= 45)
        plt.tight_layout()
        ax.set_title(col)
        st.pyplot(fig)

    # Boxplots
    st.write("📦 Box Plots (Outliers)")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        plt.xticks(rotation= 45)
        plt.tight_layout()
        ax.set_title(col)
        st.pyplot(fig)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.write("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Scatter plots
    if len(numeric_cols) >= 2:
        st.write("🔵 Scatter Plots")
        for i in range(len(numeric_cols)-1):
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=df[numeric_cols[i]],
                y=df[numeric_cols[i+1]],
                ax=ax
            )
            ax.set_title(f"{numeric_cols[i]} vs {numeric_cols[i+1]}")
            st.pyplot(fig)

    # STEP 2: DATA CLEANING
    
    st.header("🧹 STEP 2: Data Cleaning")

    df_clean = df.copy()

    # Missing values handling
    st.subheader("1️⃣ Handle Missing Values")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ["int64", "float64"]:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    st.success("Missing values handled")

    # Remove duplicates
    st.subheader("2️⃣ Remove Duplicates")
    before = df_clean.shape[0]
    df_clean.drop_duplicates(inplace=True)
    after = df_clean.shape[0]
    st.write(f"Removed {before - after} duplicate rows")

    # Fix inconsistent categories
    st.subheader("3️⃣ Fix Inconsistent Categories")
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()

    st.success("Categorical inconsistencies fixed")

    # Outlier handling (IQR)
    st.subheader("4️⃣ Handle Outliers (IQR Method)")
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    st.success("Outliers handled using IQR")

    # STEP 3: DATA PREPROCESSING
    
    st.header("⚙️ STEP 3: Data Pre-Processing (ML Ready)")

    df_processed = df_clean.copy()

    # Encoding
    st.subheader("1️⃣ Encoding Categorical Variables")
    encoding_type = st.selectbox("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])

    if encoding_type == "Label Encoding":
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    else:
        df_processed = pd.get_dummies(df_processed, drop_first=True)

    # Feature Scaling
    st.subheader("2️⃣ Feature Scaling")
    scaler_type = st.selectbox("Choose Scaling Method", ["StandardScaler", "MinMaxScaler"])

    scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
    num_cols = df_processed.select_dtypes(include=["int64", "float64"]).columns
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

    st.success("Feature scaling applied")

    # Feature Selection (Low Variance)
    st.subheader("3️⃣ Feature Selection")

# Select only numeric columns
    numeric_df = df_processed.select_dtypes(include=["int64", "float64"])

# Variance calculation
    variance = numeric_df.var()

# Threshold
    selected_features = variance[variance > 0.01].index

# Keep selected numeric + all non-numeric columns
    df_processed = pd.concat(
    [numeric_df[selected_features],
     df_processed.drop(columns=numeric_df.columns)],
    axis=1
)

    st.success("Low variance numeric features removed safely")

    # FINAL OUTPUT
    
    st.header("✅ Final ML-Ready Dataset")
    st.dataframe(df_processed.head())

    csv = df_processed.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download ML Ready Dataset",
        csv,
        "ml_ready_dataset.csv",
        "text/csv"
    )

    st.balloons()