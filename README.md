# 🫀 Heart Disease Prediction

This project predicts the presence of **heart disease** in patients using **Machine Learning (Random Forest Classifier)**.  
It takes patient health details such as age, gender, cholesterol, and blood pressure as input, and predicts whether the patient is likely to have heart disease.

---

## 📂 Project Structure
```
Heart-Disease-Prediction/
│── Disease_Predictor.ipynb    → Main Jupyter Notebook with full code
│── requirements.txt            → Python dependencies
│── README.md                   → Project documentation
│── data/
│    └── Heart_user_template.csv → Sample user input template
│── models/
│    ├── heart_rf_model.pkl      → Saved Random Forest model
│    └── heart_scaler.pkl        → Saved Scaler for preprocessing
```

---

## ⚙️ Technologies Used
- Python 3  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Joblib  

---

## 🚀 How to Run the Project

1. **Clone this repository**  
   ```bash
   git clone https://github.com/sudhakargovindasamy/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**  
   ```bash
   jupyter notebook Disease_Predictor.ipynb
   ```

4. **Make Predictions using the saved model**  
   ```python
   import joblib, pandas as pd

   # Load scaler and model
   scaler = joblib.load("models/heart_scaler.pkl")
   model = joblib.load("models/heart_rf_model.pkl")

   # Load sample input
   user_data = pd.read_csv("data/Heart_user_template.csv")

   # Scale and predict
   prediction = model.predict(scaler.transform(user_data))
   print("Prediction:", prediction)
   ```

---

## 📊 Dataset
- The dataset contains patient health records with features like **age, sex, cholesterol, blood pressure, chest pain type, etc.**  
- A sample template file `Heart_user_template.csv` is included to demonstrate the expected input format.  

---

## ✅ Results
- Random Forest achieved high accuracy in predicting heart disease.  
- Feature importance analysis shows **cholesterol, age, and blood pressure** are the most influential features.  

---

## 🔮 Future Work
- Deploy the model as a **web app** using Flask/Streamlit.  
- Improve accuracy with **deep learning models**.  
- Integrate a **real-time hospital alert system**.  

---

## 👨‍💻 Author
- Sudhakar Govindasamy  
- [LinkedIn Profile](https://www.linkedin.com/in/sudhakargovindasamy) | [GitHub Profile](https://github.com/sudhakargovindasamy)
