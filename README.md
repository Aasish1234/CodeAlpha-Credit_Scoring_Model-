# CodeAlpha-Credit_Scoring_Model-
Credit Risk Prediction using Machine Learning with Explainable AI
This project demonstrates how to predict loan default risk using machine learning models. We preprocess the dataset, handle imbalanced data using SMOTE, train multiple models, and interpret predictions using LIME (Local Interpretable Model-agnostic Explanations).

📂 Dataset
Source: Kaggle Dataset - Credit Risk Dataset

Features include:

Personal details (e.g., income, employment length, home ownership)

Loan-specific information (e.g., amount, interest rate, purpose)

Credit history (e.g., credit length, prior defaults)

Target variable: loan_status

0: No Default

1: Default

📊 Problem Statement
Financial institutions need to assess the creditworthiness of loan applicants. The goal is to build a classification model that predicts whether a loan applicant will default on their loan or not.

🧹 Step 1: Data Preprocessing
Missing Values: Filled using df.fillna(0)

Categorical Encoding: Used pd.get_dummies for one-hot encoding

Feature Scaling: Applied StandardScaler to normalize numerical features

Target Variable: loan_status

python
Copy
Edit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
⚖️ Step 2: Handling Class Imbalance with SMOTE
Many real-world datasets suffer from class imbalance. We used SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes:

python
Copy
Edit
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
🤖 Step 3: Model Training
Trained three models for comparative analysis:

Logistic Regression

Decision Tree

Random Forest

python
Copy
Edit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

lr.fit(X_resampled, y_resampled)
dt.fit(X_resampled, y_resampled)
rf.fit(X_resampled, y_resampled)
📈 Step 4: Model Evaluation
Metrics:

Classification Report (Precision, Recall, F1-Score)

ROC-AUC Score

python
Copy
Edit
from sklearn.metrics import classification_report, roc_auc_score

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"---{name}---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.2f}")
🔍 Step 5: Explainability with LIME
To understand why a model predicted default or not, we use LIME to explain individual predictions from the Random Forest model:

python
Copy
Edit
import lime
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=X.columns,
    class_names=['No Default', 'Default'],
    mode='classification'
)

lime_exp = lime_explainer.explain_instance(
    X_test[i],
    rf.predict_proba,
    num_features=10
)

lime_exp.show_in_notebook(show_table=True)
📌 This step is crucial for model transparency, especially in financial applications where decision accountability matters.

📌 Results Summary
Model	Accuracy	ROC-AUC	Notes
Logistic Regression	~X.XX	~Y.YY	Baseline model
Decision Tree	~X.XX	~Y.YY	Simple, interpretable model
Random Forest	~X.XX	~Y.YY	Best performing, robust

(Replace X.XX and Y.YY with actual scores from your results)

📚 Future Improvements
Hyperparameter tuning (GridSearchCV)

Use of more advanced models (e.g., XGBoost, LightGBM)

Feature selection techniques

Model calibration for probability estimates

👨‍💻 Author
Aasish Shrestha
Machine Learning & AI Researcher | Kaggle Enthusiast
