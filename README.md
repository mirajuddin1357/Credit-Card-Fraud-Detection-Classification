<center><h2><span style="font-weight:bolder; color: red; font-size:120%;">💳 Credit Card Fraud Detection Using Machine Learning Algorithms (Classification)</span></h2></center>

---

<div style="
  text-align:center;
  background: linear-gradient(135deg, #000000, #1a1a1a);
  border:2px solid #ff0800ff;
  border-radius:18px;
  padding:25px;
  box-shadow:0 0 25px rgba(255,8,0,0.3);
  transition:all 0.4s ease;
">
  <img src="image.png"
       width="680"
       style="
         border-radius:15px;
         box-shadow:0 0 35px rgba(255,8,0,0.4);
         transition: transform 0.4s ease, box-shadow 0.4s ease;
       "
  >
  <p style="
     color:#FFD700;
     font-size:20px;
     font-family:'Poppins', sans-serif;
     font-weight:600;
     margin-top:12px;
     letter-spacing:1px;
     text-shadow:0 0 10px rgba(255,8,0,0.8);
  ">
  Detecting fraudulent credit card transactions with precision engineering and high-performance AI.
  </p>
</div>


---

<div
  style="border-radius:10px; border:#8B0000 solid; padding:15px; background-color:#FAF3F3; font-size:100%; text-align:left; color:#222;">
  <h3 align="left" style="color:#8B0000;">Project Overview</h3>
  <p style="margin:6px 0 0; line-height:1.6;">
    This project focuses on <strong>detecting fraudulent credit card transactions</strong> using advanced <strong>Machine Learning algorithms</strong>. 
    The dataset contains transactions made by European cardholders in September 2013. Out of 284,807 transactions, only 492 are frauds, making the dataset <strong>highly imbalanced</strong> (0.172%).
    <br><br>
    We aim to build a reliable system to clean the data, use <strong>SMOTE</strong> to balance the groups, and use <strong>different models</strong> to catch fraud accurately while keeping mistakes low  using multiple <strong>Machine Learning classification algorithms</strong>.
  </p>
</div>

---

<div
    style="border-radius:15px; border:#8B0000 solid; padding:10px; background-color:#FAF3F3; font-size:100%; text-align:left; color:#222;">
    Used dataset: <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">Credit Card Fraud Detection Dataset (Kaggle)</a>
</div>

<div class="container">
    <h3 style='color:#8B0000; text-align:center;'>Transaction Dataset Attributes</h3>
    <table style='border-radius:10px; border:#8B0000 solid; width:100%; border-collapse:collapse; margin-top:10px;'>
          <tr style='background-color:white; color:black; border-bottom:2px solid rgba(236, 243, 243, 1);'>
            <th style='padding:10px;'>Attribute</th>
            <th style='padding:10px;'>Description</th>
            <th style='padding:10px;'>Type</th>
          </tr>
          <tr style='border-bottom:1px solid rgba(244, 243, 243, 1);'>
            <td style='padding:8px;'>Time</td>
            <td style='padding:8px;'>Seconds passed between this and the first transaction.</td>
            <td style='padding:8px;'>Numeric</td>
          </tr>
          <tr style='border-bottom:1px solid rgba(244, 243, 243, 1);'>
            <td style='padding:8px;'>V1 - V28</td>
            <td style='padding:8px;'>V1 - V28 These are 28 numerical features generated through Principal Component Analysis (PCA). Due to privacy and confidentiality concerns/privacy issues, the original personal data (such as location, merchant, or cardholder details) has been transformed into these abstract/numeric variables while protecting the underlying/hidden statistical relationships.</td>
            <td style='padding:8px;'>Numeric</td>
          </tr>
          <tr style='border-bottom:1px solid rgba(244, 243, 243, 1);'>
            <td style='padding:8px;'>Amount</td>
            <td style='padding:8px;'>Transaction amount.</td>
            <td style='padding:8px;'>Numeric</td>
          </tr>
          <tr style='border-bottom:1px solid rgba(244, 243, 243, 1);'>  
            <td style='padding:8px;'>Class</td>
            <td style='padding:8px;'>1 for Fraud, 0 for Legitimate/Normal. (Target)</td>
            <td style='padding:8px;'>Boolean</td>
          </tr>
    </table>
</div>

---

### Project Structure
```text
📁 Credit-Card-Fraud-Detection/
│
├── creditcard.csv                 # Original dataset (Kaggle)
├── image.png                      # Project banner (AI Generated)
├── credit-card-fraud-detection.ipynb # Main training notebook
├── app.py                         # Futuristic Streamlit Web App
├── fraud_detection_model.pkl      # Trained model file
└── README.md                      # Documentation
```

---

### Tools & Libraries Used
1. **Python:** Programming language for the project.  
2. **NumPy & Pandas:** Data handling and manipulation.  
3. **Matplotlib, Plotly & Seaborn:** Data visualization.  
4. **Scikit-learn:** Preprocessing, train-test split, ML algorithms, and evaluation metrics.  
5. **XGBoost, LightGBM, CatBoost:** Advanced classification models.  
6. **Imbalanced-learn (SMOTE):** Handling imbalanced datasets.  
7. **Optuna:** Hyperparameter optimization.  
8. **Jupyter Notebook / VS Code:** Development environment.  
9. **Git & GitHub:** Version control and collaboration.
10. **LFS:** Large File Storage for storing large files (e.g., `creditcard.csv`) to upload on GitHub through Git.
11. **Streamlit:** Web app development.

---

### Project Workflow
1. **Import Libraries:** Setting up the environment.
2. **Data Collection:** Loading the European cardholders dataset.
3. **Data Preprocessing:** Cleaning, scaling (StandardScaler).
4. **EDA:** Comprehensive univariate, bivariate, and multivariate analysis including class-wise distributions, correlation matrices, and boxplots for outlier detection.
5. **Feature Engineering:** Log transforms and feature selection.
6. **Balancing:** SMOTE implementation.
7. **Model Selection:** Performance benchmarking.
8. **Deployment:** Streamlit app integration.

---

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">Accuracy</th>
      <th align="center">Precision</th>
      <th align="center">Recall</th>
      <th align="center">F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong style="color: red;">XGBoost</strong></td>
      <td align="center"><strong style="color: red;">0.9997</strong></td>
      <td align="center">0.9995</td>
      <td align="center"><strong style="color: red;">1.0000</strong></td>
      <td align="center"><strong style="color: red;">0.9997</strong></td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td align="center">0.9997</td>
      <td align="center"><strong>0.9996</strong></td>
      <td align="center">0.9999</td>
      <td align="center">0.9997</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td align="center">0.9997</td>
      <td align="center">0.9996</td>
      <td align="center">0.9999</td>
      <td align="center">0.9997</td>
    </tr>
    <tr>
      <td>CatBoost</td>
      <td align="center">0.9995</td>
      <td align="center">0.9991</td>
      <td align="center">1.0000</td>
      <td align="center">0.9995</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td align="center">0.9807</td>
      <td align="center">0.9907</td>
      <td align="center">0.9704</td>
      <td align="center">0.9805</td>
    </tr>
  </tbody>
</table>

---

Live Streamlit App: [Credit Card Fraud Detection](https://credit-card-fraud-detection-classification.streamlit.app/)
**Built by Miraj Ud Din** | [Portfolio](https://miraj-portfolio-db-2026.web.app/)
  
---
