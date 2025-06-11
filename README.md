# Real-Time-Water-Quality-Prediction-App
# 💧 Real-Time Water Quality Prediction App

This web application allows users to **predict water potability** in real-time based on various chemical and physical parameters. The model is trained using supervised machine learning algorithms, and the app interface is built using **Streamlit**.

## 🚀 Features

- 🧠 Machine Learning-based prediction using **CatBoost** and **Random Forest**
- 📊 Input parameters like pH, hardness, solids, chloramines, etc.
- ✅ Real-time prediction on whether the water is **safe to drink** or not
- 🌐 User-friendly web interface built with **Streamlit**
- 📉 Displays confidence of prediction for better interpretability

---


---

## 🛠 Technologies Used

- **Python**
- **Pandas**, **NumPy**, **Scikit-learn**
- **CatBoost**, **Random Forest**
- **Streamlit** (Web UI)
- **Joblib/Pickle** for model serialization

---

## 📈 ML Model Details

The application uses:
- **CatBoost Classifier** for handling categorical features efficiently.
- **Random Forest Classifier** as a backup or comparison.
- Trained on the popular **Water Potability Dataset**.
- Evaluated using **Accuracy**, **G-Mean**, and **AUC** metrics.

---

## ✅ How to Run the App Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/water-quality-prediction.git
   cd water-quality-prediction
2.Install dependencies
    pip install -r requirements.txt
3.Launch the app:
   streamlit run app.py

🔍 Input Parameters
Feature                   	Description
pH	                        Acidity or alkalinity of water
Hardness	                  Concentration of calcium and magnesium
Solids	                     Total dissolved solids in ppm
Chloramines                  Disinfectant used in water treatment
Sulfate	                    Sulfate concentration
Conductivity               	Water's electrical conductivity
Organic_carbon	           Level of organic pollutants
Trihalomethanes	            Disinfection byproducts
Turbidity	                 Clarity of the water

Screenshots
![image](https://github.com/user-attachments/assets/654a04b6-5d9e-4531-884f-ff351c10df1d)

![image](https://github.com/user-attachments/assets/02c20113-dd84-4812-8b0d-a3d4f5be5220)
![image](https://github.com/user-attachments/assets/69b5fa69-e591-434b-9ec7-494143d87eea)

