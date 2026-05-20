# Disease Risk Grouping and Diagnosis Prediction Using ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

# CONFIGURATION
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# --- 1. BACKEND: SILENT ENGINE START ---
try:
    df = pd.read_csv('/content/sample_data/heart_disease_uci.csv')

    # Preprocessing
    df['Sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['High_Fasting_BloodSugar'] = df['fbs'].map({True: 1, False: 0})
    df.rename(columns={'age': 'Age', 'trestbps': 'BloodPressure', 'chol': 'Cholesterol',
                       'thalch': 'MaxHeartRate', 'num': 'HeartDisease_Stage'}, inplace=True)

    feature_cols = ['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHeartRate', 'High_Fasting_BloodSugar']
    imputer = SimpleImputer(strategy='mean')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # Targets
    df['Has_HeartDisease'] = df['HeartDisease_Stage'].apply(lambda x: 1 if x > 0 else 0)
    df['Has_Hypertension'] = df['BloodPressure'].apply(lambda x: 1 if x > 130 else 0)
    df['Has_Diabetes_Risk'] = df['High_Fasting_BloodSugar'].astype(int)

    # Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    # Unsupervised (Clusters)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Patient_Profile_Cluster'] = kmeans.fit_predict(X_scaled)

    # Supervised (Prediction)
    X = df[feature_cols + ['Patient_Profile_Cluster']]
    y = df[['Has_HeartDisease', 'Has_Hypertension', 'Has_Diabetes_Risk']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    multi_target_model = MultiOutputClassifier(rf, n_jobs=-1)
    multi_target_model.fit(X_train, y_train)

except Exception:
    print("Error: Dataset not found. Please upload 'heart_disease_uci.csv'.")
    raise SystemExit

# --- 2. FRONTEND: INPUT INTERFACE ---
def get_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            pass

print("\nCLINICAL ENTRY PORTAL")
print("---------------------")

try:
    # Inputs
    p_age = get_input("Age: ")
    s_in = input("Sex (M/F): ").strip().upper()
    p_sex = 1 if s_in.startswith('M') else 0
    p_bp = get_input("Blood Pressure: ")
    p_chol = get_input("Cholesterol: ")
    p_hr = get_input("Max Heart Rate: ")
    f_in = input("Fasting Blood Sugar > 120? (Y/N): ").strip().upper()
    p_fbs = 1 if f_in.startswith('Y') else 0

    # Processing
    input_data = pd.DataFrame([[p_age, p_sex, p_bp, p_chol, p_hr, p_fbs]], columns=feature_cols)
    filled = pd.DataFrame(imputer.transform(input_data), columns=feature_cols)
    scaled = scaler.transform(filled)

    cluster_id = kmeans.predict(scaled)[0]
    filled['Patient_Profile_Cluster'] = cluster_id
    preds = multi_target_model.predict(filled)[0]

    # Text Report
    print("\nDIAGNOSTIC SUMMARY")
    print("------------------")
    print(f"Risk Profile Group: {cluster_id}")
    labels = ['Heart Disease', 'Hypertension', 'Diabetes Risk']
    for label, result in zip(labels, preds):
        status = "POSITIVE (Risk Detected)" if result == 1 else "Negative (Healthy)"
        print(f"{label:<20} : {status}")

    # --- 3. VISUALIZATION: THE 2x2 DASHBOARD ---
    print("\nGenerating Complete Insight Dashboard...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    plt.suptitle(f"Medical Insight Dashboard | Patient Age: {int(p_age)} | Sex: {s_in}", fontsize=16, fontweight='bold')

    # [TOP-LEFT] PATIENT CONTEXT (Where do they fit?)
    sns.scatterplot(data=df, x='Age', y='MaxHeartRate', hue='Patient_Profile_Cluster',
                    palette='viridis', alpha=0.4, ax=axs[0,0], legend='full')
    axs[0,0].scatter(p_age, p_hr, color='red', s=300, marker='*', label='THIS PATIENT', zorder=10)
    axs[0,0].set_title("1. Patient Profile Context (Cluster Analysis)", fontweight='bold')
    axs[0,0].legend(loc='upper right')

    # [TOP-RIGHT] RELIABILITY CHECK (Confusion Matrix for Heart Disease)
    # This answers: "How often is the AI right vs wrong?"
    y_pred_heart = multi_target_model.estimators_[0].predict(X_test)
    cm = confusion_matrix(y_test['Has_HeartDisease'], y_pred_heart)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0,1], cbar=False, annot_kws={"size": 14})
    axs[0,1].set_title("2. AI Reliability (Confusion Matrix - Heart Disease)", fontweight='bold')
    axs[0,1].set_xlabel("Predicted Diagnosis")
    axs[0,1].set_ylabel("True Diagnosis")
    axs[0,1].set_xticklabels(['Healthy', 'Sick'])
    axs[0,1].set_yticklabels(['Healthy', 'Sick'])

    # [BOTTOM-LEFT] EXPLAINABILITY (Feature Importance)
    # This answers: "What factors mattered most?"
    importances = np.mean([est.feature_importances_ for est in multi_target_model.estimators_], axis=0)
    indices = np.argsort(importances)
    axs[1,0].barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
    axs[1,0].set_yticks(range(len(indices)))
    axs[1,0].set_yticklabels([X.columns[i] for i in indices])
    axs[1,0].set_title("3. Top Clinical Risk Factors (AI Logic)", fontweight='bold')
    axs[1,0].set_xlabel("Impact Score")

    # [BOTTOM-RIGHT] CONFIDENCE (Accuracy Bars)
    # This answers: "How accurate is it for EACH disease?"
    y_pred = multi_target_model.predict(X_test)
    accuracies = [accuracy_score(y_test[col], y_pred[:, i]) for i, col in enumerate(y.columns)]
    sns.barplot(x=labels, y=accuracies, palette=['#3498db', '#e74c3c', '#f1c40f'], ax=axs[1,1])
    axs[1,1].set_title("4. Overall System Confidence", fontweight='bold')
    axs[1,1].set_ylim(0, 1.15)
    for i, v in enumerate(accuracies):
        axs[1,1].text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit main title
    plt.show()

except Exception as e:
    print(f"Processing Error: {e}")
