import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Healthcare Dataset Implementation
class HealthcarePredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess healthcare dataset"""
        # For demonstration, creating synthetic data similar to the Kaggle dataset
        np.random.seed(42)
        n_samples = 200000
        
        data = {
            'Age': np.random.randint(18, 90, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Blood_Type': np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], n_samples),
            'Medical_Condition': np.random.choice(['Diabetes', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis'], n_samples),
            'Admission_Type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
            'Insurance_Provider': np.random.choice(['Aetna', 'Blue Cross', 'Cigna', 'Medicare', 'UnitedHealthcare'], n_samples),
            'Billing_Amount': np.random.uniform(1000, 50000, n_samples),
            'Room_Number': np.random.randint(100, 500, n_samples),
            'Days_in_Hospital': np.random.randint(1, 30, n_samples),
            'Test_Results': np.random.choice(['Normal', 'Abnormal', 'Inconclusive'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable: High Risk Patient (based on multiple factors)
        # Create risk probabilities based on multiple factors
        risk_prob = 0.1  # Base risk
        risk_prob += 0.3 * (df['Age'] > 65)
        risk_prob += 0.2 * (df['Days_in_Hospital'] > 10) 
        risk_prob += 0.25 * (df['Billing_Amount'] > 30000)
        risk_prob += 0.2 * (df['Test_Results'] == 'Abnormal')

        # Add noise and create probabilistic target
        np.random.seed(42)
        df['High_Risk'] = np.random.binomial(1, np.clip(risk_prob, 0, 1), len(df))
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        # Separate features and target
        X = df.drop(['High_Risk'], axis=1)
        y = df['High_Risk']
        
        # Handle categorical variables
        categorical_cols = ['Gender', 'Blood_Type', 'Medical_Condition', 'Admission_Type', 'Insurance_Provider', 'Test_Results']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_and_evaluate(self, X, y):
        """Train model and evaluate performance"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'actual': y_test
        }

# Cybersecurity Dataset Implementation
class CybersecurityPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """Load and preprocess cybersecurity dataset"""
        # Creating synthetic data similar to CSE-CIC-IDS2018 dataset
        np.random.seed(42)
        n_samples = 200000
        
        # Generate network traffic features
        data = {
            'Flow_Duration': np.random.exponential(1000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(50, n_samples),
            'Total_Backward_Packets': np.random.poisson(30, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.exponential(2000, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.exponential(1500, n_samples),
            'Fwd_Packet_Length_Max': np.random.gamma(2, 100, n_samples),
            'Fwd_Packet_Length_Min': np.random.exponential(50, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(200, 50, n_samples),
            'Bwd_Packet_Length_Max': np.random.gamma(2, 80, n_samples),
            'Bwd_Packet_Length_Min': np.random.exponential(40, n_samples),
            'Flow_Bytes_per_Second': np.random.exponential(5000, n_samples),
            'Flow_Packets_per_Second': np.random.exponential(10, n_samples),
            'Flow_IAT_Mean': np.random.exponential(100, n_samples),
            'Flow_IAT_Std': np.random.exponential(200, n_samples),
            'Flow_IAT_Max': np.random.exponential(500, n_samples),
            'Fwd_IAT_Total': np.random.exponential(1000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(100, n_samples),
            'Bwd_IAT_Total': np.random.exponential(800, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(80, n_samples),
            'PSH_Flag_Count': np.random.poisson(2, n_samples),
            'URG_Flag_Count': np.random.poisson(0.1, n_samples),
            'Average_Packet_Size': np.random.normal(150, 30, n_samples),
            'Avg_Fwd_Segment_Size': np.random.normal(120, 25, n_samples),
            'Avg_Bwd_Segment_Size': np.random.normal(100, 20, n_samples)
        }
        
        df = pd.DataFrame(data)
    
        # Calculate attack probability based on features
        attack_prob = 0.05  # Base attack probability
        attack_prob += 0.2 * (df['Flow_Packets_per_Second'] > df['Flow_Packets_per_Second'].quantile(0.9))
        attack_prob += 0.25 * (df['Flow_Bytes_per_Second'] > df['Flow_Bytes_per_Second'].quantile(0.95))
        attack_prob += 0.15 * (df['PSH_Flag_Count'] > 5)
        attack_prob += 0.2 * (df['URG_Flag_Count'] > 1)
        attack_prob += 0.1 * (df['Flow_Duration'] > df['Flow_Duration'].quantile(0.95))

        # Assign attack types
        attack_types = ['BENIGN', 'DoS', 'DDoS', 'BruteForce', 'Web Attack', 'Infiltration', 'Botnet']

        # Create probabilistic labels with noise
        np.random.seed(42)
        is_attack = np.random.binomial(1, np.clip(attack_prob, 0, 1), len(df))
        labels = np.where(is_attack, np.random.choice(attack_types[1:], len(df)), 'BENIGN')     
        df['Label'] = labels
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        # Separate features and target
        X = df.drop(['Label'], axis=1)
        y = df['Label']
        
        # Convert to binary classification (BENIGN vs ATTACK)
        y_binary = (y != 'BENIGN').astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y_binary, y
    
    def train_and_evaluate(self, X, y_binary, y_multi):
        """Train model and evaluate performance"""
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
        
        # Train binary classification model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'actual': y_test
        }

# Execute Healthcare Model
print("=== HEALTHCARE PREDICTION MODEL ===")
healthcare_model = HealthcarePredictionModel()
healthcare_data = healthcare_model.load_and_preprocess_data("healthcare_dataset.csv")
X_health, y_health = healthcare_model.preprocess_features(healthcare_data)
health_results = healthcare_model.train_and_evaluate(X_health, y_health)

print(f"Healthcare Model Results:")
print(f"Accuracy: {health_results['accuracy']:.4f}")
print(f"AUC Score: {health_results['auc_score']:.4f}")
print(f"Confusion Matrix:\n{health_results['confusion_matrix']}")
print(f"Classification Report:\n{health_results['classification_report']}")

# Execute Cybersecurity Model
print("\n=== CYBERSECURITY PREDICTION MODEL ===")
cyber_model = CybersecurityPredictionModel()
cyber_data = cyber_model.load_and_preprocess_data()
X_cyber, y_cyber_binary, y_cyber_multi = cyber_model.preprocess_features(cyber_data)
cyber_results = cyber_model.train_and_evaluate(X_cyber, y_cyber_binary, y_cyber_multi)

print(f"Cybersecurity Model Results:")
print(f"Accuracy: {cyber_results['accuracy']:.4f}")
print(f"AUC Score: {cyber_results['auc_score']:.4f}")
print(f"Confusion Matrix:\n{cyber_results['confusion_matrix']}")
print(f"Classification Report:\n{cyber_results['classification_report']}")
