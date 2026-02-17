# 🏠 House Price Prediction System

A production-ready Machine Learning application that predicts real estate prices using a trained ML pipeline, FastAPI backend, and Streamlit interactive dashboard.

This project demonstrates end-to-end ML system design including data preprocessing, model training, API deployment, frontend integration, and CI/CD automation.



# 📌 Project Summary

The objective of this project is to build an intelligent system capable of predicting housing prices based on geographic location, population statistics, income levels, and housing characteristics.

The system integrates:

• Machine Learning Pipeline  
• FastAPI Backend Service  
• Streamlit Interactive Dashboard  
• CI/CD Automation  

This project simulates a real-world production ML deployment workflow.


# 🚀 Technologies Used

Machine Learning:
- Scikit-learn
- Pandas
- NumPy

Backend:
- FastAPI
- Uvicorn

Frontend:
- Streamlit
- Plotly

DevOps & Tools:
- Git
- GitHub
- GitHub Actions (CI/CD)

Visualization:
- Plotly Gauge Charts
- Interactive Dashboard UI


# ✨ Features

• Real-time house price prediction  
• End-to-end ML pipeline with preprocessing  
• Interactive professional dashboard  
• FastAPI REST API integration  
• Model performance visualization  
• Gauge chart for price comparison  
• Modern UI/UX design  
• CI/CD automation using GitHub Actions  
• Production-ready architecture  


# ⌨️ Keyboard Shortcuts

Streamlit Dashboard:

CTRL + C → Stop server  
CTRL + R → Refresh app (browser)  
CTRL + ENTER → Run selected code (if using notebook)  


# ⚙️ Project Workflow / Process

1. Data Collection & Exploration (EDA)
2. Data Preprocessing (Missing values, encoding, scaling)
3. Model Training & Comparison
4. Cross Validation
5. Hyperparameter Tuning
6. Final Model Selection
7. Model Serialization (Pipeline)
8. FastAPI Backend Development
9. Streamlit Frontend Integration
10. CI/CD Pipeline Setup
11. Deployment Ready System


# 🤖 Machine Learning Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor (Final Model)


# 🎯 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score


# 🧠 What I Learned

• Building complete end-to-end ML systems  
• Creating reusable Scikit-learn pipelines  
• Backend API development with FastAPI  
• Professional UI dashboard design using Streamlit  
• Model deployment concepts  
• CI/CD automation using GitHub Actions  
• Debugging real-world integration issues  
• Production-level project structuring  


# 🛠️ How I Built This Project

1. Performed exploratory data analysis to understand feature relationships.
2. Built preprocessing pipelines for numerical and categorical features.
3. Trained multiple regression models and compared performance.
4. Selected the best-performing model using cross-validation.
5. Tuned hyperparameters using GridSearchCV.
6. Saved the trained pipeline for inference.
7. Created FastAPI endpoints for prediction.
8. Built an interactive Streamlit dashboard.
9. Connected frontend to backend using REST API.
10. Implemented CI/CD workflow using GitHub Actions.


# 🔮 How This Project Can Be Improved

• Deploy model on AWS / Docker containers  
• Add model monitoring and logging  
• Implement user authentication  
• Add real-time data ingestion  
• Improve feature engineering  
• Add explainable AI (SHAP / LIME)  
• Enable batch prediction support  
• Integrate database storage  


# 📁 Project Structure

house-price-prediction/
│
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD Pipeline (GitHub Actions)
│
├── model.pkl                     # Trained ML pipeline model
├── app.py                        # FastAPI backend
├── streamlit_app.py              # Streamlit frontend dashboard
├── house-price-prediction.ipynb  # Model training notebook
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignored files



# ▶️ How to Run the Project

Step 1 — Clone Repository

git clone https://github.com/vinayak533/house-price-prediction.git
cd house-price-prediction


Step 2 — Install Dependencies

pip install -r requirements.txt


Step 3 — Run FastAPI Backend

uvicorn app:app --reload


Step 4 — Run Streamlit Frontend

streamlit run streamlit_app.py


Step 5 — Open in Browser

http://localhost:8501


# 📊 System Architecture

Streamlit UI  →  FastAPI Backend  →  ML Pipeline Model  →  Prediction Output


# 👨‍💻 Developer

Vinayak K V  
Machine Learning Engineer | Data Scientist  


# ⭐ If you like this project, please give it a star!
