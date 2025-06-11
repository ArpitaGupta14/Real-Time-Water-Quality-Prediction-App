import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# === Page Configuration ===
st.set_page_config(page_title="Water Potability Classifier", layout="wide")
st.title("💧 Water Potability Classifier")

# === File Upload ===
uploaded_file = st.file_uploader("Upload your dataset (.csv or .zip)", type=["csv", "zip"])

if uploaded_file is not None:
    # === File Extraction ===
    if uploaded_file.name.endswith('.zip'):
        import zipfile, io
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            csv_name = zip_ref.namelist()[0]
            with zip_ref.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Loaded Successfully!")
    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # === Missing Values Imputation ===
    for col in ['ph', 'Sulfate', 'Trihalomethanes']:
        df[col] = df.groupby('Potability')[col].transform(lambda x: x.fillna(x.mean()))

    # === Feature & Target Split ===
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # === Train-Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # === Handle Imbalanced Data ===
    smt = SMOTE(random_state=42)
    X_train, y_train = smt.fit_resample(X_train, y_train)

    # === Scaling ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Store in session for use in prediction
    st.session_state['scaler'] = scaler
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test

    # === Model Definitions ===
    models = {
        'RandomForest': RandomForestClassifier(),
       
        'SVM': SVC(probability=True),
        
       
        'CatBoost': CatBoostClassifier(verbose=0)  # Added CatBoost
    }

    param_grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [5, 10]},
        'XGBoost': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1]},
        'SVM': {'C': [0.5, 1], 'kernel': ['rbf']},
        'GradientBoosting': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1]},
        'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]},
        'CatBoost': {'iterations': [100, 200], 'learning_rate': [0.03, 0.1], 'depth': [4, 6]}
    }

    selected_model = st.selectbox("🧠 Select a Model for Training", list(models.keys()))

    if st.button("🚀 Train & Evaluate Model"):
        with st.spinner("Training and tuning model..."):
            if selected_model == 'CatBoost':
                # CatBoost handles categorical features internally, no need for scaler
                X_train_cb = X_train
                X_test_cb = X_test
                grid = GridSearchCV(models[selected_model], param_grids[selected_model], scoring='accuracy', cv=5)
                grid.fit(X_train_cb, y_train)
            else:
                grid = GridSearchCV(models[selected_model], param_grids[selected_model], scoring='accuracy', cv=5)
                grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            st.session_state['model'] = best_model  # Save trained model

            acc = accuracy_score(y_test, y_pred)
            st.subheader(f"✅ {selected_model} Accuracy: {acc:.4f}")
            st.text("Classification Report:")
            st.code(classification_report(y_test, y_pred))

            # === Confusion Matrix ===
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{selected_model} Confusion Matrix')
            st.pyplot(fig)

    # === Prediction Form ===
    st.markdown("## 🔎 Predict Water Potability Using Your Input")
    with st.form("prediction_form"):
        ph = st.number_input("ph", min_value=0.0, max_value=14.0, step=0.1)
        Hardness = st.number_input("Hardness", min_value=0.0)
        Solids = st.number_input("Solids", min_value=0.0)
        Chloramines = st.number_input("Chloramines", min_value=0.0)
        Sulfate = st.number_input("Sulfate", min_value=0.0)
        Conductivity = st.number_input("Conductivity", min_value=0.0)
        Organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
        Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
        Turbidity = st.number_input("Turbidity", min_value=0.0)

        submit_btn = st.form_submit_button("🔍 Predict")

        if submit_btn:
            if 'model' not in st.session_state:
                st.warning("⚠️ Please train a model first by clicking 'Train & Evaluate Model'.")
            else:
                input_features = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                                            Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
                input_scaled = st.session_state['scaler'].transform(input_features)

                prediction = st.session_state['model'].predict(input_scaled)

                if prediction[0] == 1:
                    st.success("💧 The water is **Safe for Drinking**.")
                else:
                    st.error("⚠️ The water is **Not Safe for Drinking**.")
else:
    st.info("📁 Please upload a CSV or ZIP file to get started.")
