import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="ML Prediction Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = {}

class HealthcarePredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = []
        
    def generate_synthetic_data(self, n_samples, columns_config, risk_config):
        """Generate synthetic healthcare data based on user configuration"""
        np.random.seed(42)
        
        data = {}
        
        # Generate data based on user configuration
        for col, config in columns_config.items():
            if config['type'] == 'numeric':
                if col == 'Age':
                    data[col] = np.random.randint(config['min'], config['max'], n_samples)
                elif 'Amount' in col or 'Days' in col:
                    data[col] = np.random.uniform(config['min'], config['max'], n_samples)
                else:
                    data[col] = np.random.normal(config['mean'], config['std'], n_samples)
            elif config['type'] == 'categorical':
                data[col] = np.random.choice(config['values'], n_samples)
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk configuration
        risk_prob = 0.1  # Base risk
        
        for col, weight in risk_config.items():
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, use quantile-based risk
                    risk_prob += weight * (df[col] > df[col].quantile(0.7))
                else:
                    # For categorical columns, assign risk to specific categories
                    high_risk_categories = ['Abnormal', 'Emergency', 'Diabetes', 'Hypertension']
                    risk_prob += weight * df[col].isin(high_risk_categories)
        
        # Create probabilistic target
        df['High_Risk'] = np.random.binomial(1, np.clip(risk_prob, 0, 1), len(df))
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        X = df.drop(['High_Risk'], axis=1)
        y = df['High_Risk']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        self.feature_columns = X.columns.tolist()
        
        # Handle missing values and scale
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_and_evaluate(self, X, y):
        """Train model and evaluate performance"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        return results

class CybersecurityPredictionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def generate_synthetic_data(self, n_samples, columns_config, risk_config):
        """Generate synthetic cybersecurity data based on user configuration"""
        np.random.seed(42)
        
        data = {}
        
        for col, config in columns_config.items():
            if config['type'] == 'numeric':
                if 'Duration' in col:
                    if 'scale' in config:
                        data[col] = np.random.exponential(config['scale'], n_samples)
                    else:
                        # Fallback to normal distribution if scale not provided
                        mean = config.get('mean', 1000)
                        std = config.get('std', 200)
                        data[col] = np.random.normal(mean, std, n_samples)
                elif 'Packets' in col:
                    if 'lambda' in config:
                        data[col] = np.random.poisson(config['lambda'], n_samples)
                    else:
                        # Fallback to normal distribution if lambda not provided
                        mean = config.get('mean', 50)
                        std = config.get('std', 10)
                        data[col] = np.random.normal(mean, std, n_samples)
                elif 'Bytes' in col:
                    if 'scale' in config:
                        data[col] = np.random.exponential(config['scale'], n_samples)
                    else:
                        # Fallback to normal distribution if scale not provided
                        mean = config.get('mean', 2000)
                        std = config.get('std', 400)
                        data[col] = np.random.normal(mean, std, n_samples)
                else:
                    # For other numeric columns, use mean and std
                    mean = config.get('mean', 0)
                    std = config.get('std', 1)
                    data[col] = np.random.normal(mean, std, n_samples)
            elif config['type'] == 'categorical':
                data[col] = np.random.choice(config['values'], n_samples)

        
        df = pd.DataFrame(data)
        
        # Create attack probability based on risk configuration
        attack_prob = 0.05  # Base attack probability
        
        for col, weight in risk_config.items():
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                attack_prob += weight * (df[col] > df[col].quantile(0.9))
        
        # Create target variable
        attack_types = ['BENIGN', 'DoS', 'DDoS', 'BruteForce', 'Web Attack', 'Infiltration', 'Botnet']
        is_attack = np.random.binomial(1, np.clip(attack_prob, 0, 1), len(df))
        labels = np.where(is_attack, np.random.choice(attack_types[1:], len(df)), 'BENIGN')
        df['Label'] = labels
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        X = df.drop(['Label'], axis=1)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        self.feature_columns = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_and_evaluate(self, X, y):
        """Train model and evaluate performance"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        return results

def home_page():
    """Main landing page for model selection"""
    st.title("🤖 Machine Learning Prediction Models")
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3>Choose Your Prediction Model</h3>
        <p>Select the type of prediction model you want to train and use</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🏥 Healthcare Prediction Model")
        st.write("Predict high-risk patients based on medical data including age, medical conditions, billing amounts, and test results.")
        if st.button("🏥 Healthcare Model", key="healthcare", use_container_width=True):
            st.session_state.page = 'healthcare_config'
            st.session_state.model_type = 'healthcare'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🔒 Cybersecurity Threat Prediction")
        st.write("Detect network attacks and threats based on network traffic patterns, packet analysis, and flow characteristics.")
        if st.button("🔒 Cybersecurity Model", key="cybersecurity", use_container_width=True):
            st.session_state.page = 'cybersecurity_config'
            st.session_state.model_type = 'cybersecurity'
            st.rerun()

def healthcare_config_page():
    """Configuration page for healthcare model"""
    st.title("🏥 Healthcare Model Configuration")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    # Default healthcare columns configuration
    default_healthcare_config = {
        'Age': {'type': 'numeric', 'min': 18, 'max': 90},
        'Gender': {'type': 'categorical', 'values': ['Male', 'Female']},
        'Blood_Type': {'type': 'categorical', 'values': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']},
        'Medical_Condition': {'type': 'categorical', 'values': ['Diabetes', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis']},
        'Admission_Type': {'type': 'categorical', 'values': ['Emergency', 'Elective', 'Urgent']},
        'Insurance_Provider': {'type': 'categorical', 'values': ['Aetna', 'Blue Cross', 'Cigna', 'Medicare', 'UnitedHealthcare']},
        'Billing_Amount': {'type': 'numeric', 'min': 1000, 'max': 50000},
        'Days_in_Hospital': {'type': 'numeric', 'min': 1, 'max': 30},
        'Test_Results': {'type': 'categorical', 'values': ['Normal', 'Abnormal', 'Inconclusive']}
    }
    
    default_risk_config = {
        'Age': 0.3,
        'Days_in_Hospital': 0.2,
        'Billing_Amount': 0.25,
        'Test_Results': 0.2
    }
    
    st.subheader("📊 Dataset Configuration")
    
    # Dataset size
    n_samples = st.number_input("Number of samples to generate", min_value=1000, max_value=500000, value=50000, step=1000)
    
    # Column configuration
    st.subheader("🔧 Column Configuration")
    
    columns_config = {}
    
    for col, config in default_healthcare_config.items():
        with st.expander(f"Configure {col}"):
            include_col = st.checkbox(f"Include {col}", value=True, key=f"include_{col}")
            
            if include_col:
                if config['type'] == 'numeric':
                    min_val = st.number_input(f"Min value for {col}", value=config['min'], key=f"min_{col}")
                    max_val = st.number_input(f"Max value for {col}", value=config['max'], key=f"max_{col}")
                    columns_config[col] = {'type': 'numeric', 'min': min_val, 'max': max_val}
                else:
                    values = st.text_area(f"Values for {col} (comma-separated)", 
                                        value=', '.join(config['values']), key=f"values_{col}")
                    columns_config[col] = {'type': 'categorical', 'values': [v.strip() for v in values.split(',')]}
    
    # Risk configuration
    st.subheader("⚠️ Risk Parameters Configuration")
    st.write("Set the risk probability weights for each column (0.0 to 1.0)")
    
    risk_config = {}
    for col in columns_config.keys():
        if col in default_risk_config:
            weight = st.slider(f"Risk weight for {col}", 0.0, 1.0, default_risk_config[col], 0.05, key=f"risk_{col}")
            risk_config[col] = weight
    
    # Training button
    if st.button("🚀 Train Healthcare Model", use_container_width=True):
        with st.spinner("Training healthcare model..."):
            model = HealthcarePredictionModel()
            
            # Generate data
            data = model.generate_synthetic_data(n_samples, columns_config, risk_config)
            
            # Preprocess and train
            X, y = model.preprocess_features(data)
            results = model.train_and_evaluate(X, y)
            
            # Store model and results
            st.session_state.trained_model = model
            st.session_state.model_data = {
                'type': 'healthcare',
                'results': results,
                'columns_config': columns_config,
                'sample_data': data.head()
            }
            
            st.session_state.page = 'results'
            st.rerun()

def cybersecurity_config_page():
    """Configuration page for cybersecurity model"""
    st.title("🔒 Cybersecurity Model Configuration")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    # Default cybersecurity columns configuration
    default_cyber_config = {
        'Flow_Duration': {'type': 'numeric', 'scale': 1000},
        'Total_Fwd_Packets': {'type': 'numeric', 'lambda': 50},
        'Total_Backward_Packets': {'type': 'numeric', 'lambda': 30},
        'Total_Length_of_Fwd_Packets': {'type': 'numeric', 'scale': 2000},
        'Total_Length_of_Bwd_Packets': {'type': 'numeric', 'scale': 1500},
        'Flow_Bytes_per_Second': {'type': 'numeric', 'scale': 5000},
        'Flow_Packets_per_Second': {'type': 'numeric', 'scale': 10},
        'Flow_IAT_Mean': {'type': 'numeric', 'scale': 100},
        'PSH_Flag_Count': {'type': 'numeric', 'lambda': 2},
        'URG_Flag_Count': {'type': 'numeric', 'lambda': 0.1},
        'Average_Packet_Size': {'type': 'numeric', 'mean': 150, 'std': 30}
    }
    
    default_risk_config = {
        'Flow_Packets_per_Second': 0.2,
        'Flow_Bytes_per_Second': 0.25,
        'PSH_Flag_Count': 0.15,
        'URG_Flag_Count': 0.2,
        'Flow_Duration': 0.1
    }
    
    st.subheader("📊 Dataset Configuration")
    
    # Dataset size
    n_samples = st.number_input("Number of samples to generate", min_value=1000, max_value=500000, value=50000, step=1000)
    
    # Column configuration
    st.subheader("🔧 Column Configuration")
    
    columns_config = {}
    
    for col, config in default_cyber_config.items():
        with st.expander(f"Configure {col}"):
            include_col = st.checkbox(f"Include {col}", value=True, key=f"include_{col}")
            
            if include_col:
                if 'scale' in config:
                    scale = st.number_input(f"Scale parameter for {col}", value=config['scale'], key=f"scale_{col}")
                    columns_config[col] = {'type': 'numeric', 'scale': scale}
                elif 'lambda' in config:
                    lambda_val = st.number_input(f"Lambda parameter for {col}", value=config['lambda'], key=f"lambda_{col}")
                    columns_config[col] = {'type': 'numeric', 'lambda': lambda_val}
                else:
                    mean = st.number_input(f"Mean for {col}", value=config['mean'], key=f"mean_{col}")
                    std = st.number_input(f"Std for {col}", value=config['std'], key=f"std_{col}")
                    columns_config[col] = {'type': 'numeric', 'mean': mean, 'std': std}
    
    # Risk configuration
    st.subheader("⚠️ Attack Risk Parameters")
    st.write("Set the attack probability weights for each column (0.0 to 1.0)")
    
    risk_config = {}
    for col in columns_config.keys():
        if col in default_risk_config:
            weight = st.slider(f"Attack weight for {col}", 0.0, 1.0, default_risk_config[col], 0.05, key=f"risk_{col}")
            risk_config[col] = weight
    
    # Training button
    if st.button("🚀 Train Cybersecurity Model", use_container_width=True):
        with st.spinner("Training cybersecurity model..."):
            model = CybersecurityPredictionModel()
            
            # Generate data
            data = model.generate_synthetic_data(n_samples, columns_config, risk_config)
            
            # Preprocess and train
            X, y = model.preprocess_features(data)
            results = model.train_and_evaluate(X, y)
            
            # Store model and results
            st.session_state.trained_model = model
            st.session_state.model_data = {
                'type': 'cybersecurity',
                'results': results,
                'columns_config': columns_config,
                'sample_data': data.head()
            }
            
            st.session_state.page = 'results'
            st.rerun()

def results_page():
    """Display training results and provide prediction interface"""
    st.title("📈 Training Results & Prediction Interface")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.session_state.trained_model is None:
        st.error("No trained model found. Please train a model first.")
        return
    
    results = st.session_state.model_data['results']
    model_type = st.session_state.model_data['type']
    
    st.markdown("---")
    
    # Display results
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    
    with col2:
        st.metric("AUC Score", f"{results['auc_score']:.4f}")
    
    with col3:
        st.metric("Model Type", model_type.title())
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    importance_df = pd.DataFrame(list(results['feature_importance'].items()), 
                                columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    st.pyplot(fig)
    
    # Classification Report
    with st.expander("📋 Detailed Classification Report"):
        st.text(results['classification_report'])
    
    # Sample Data
    with st.expander("📄 Sample Generated Data"):
        st.dataframe(st.session_state.model_data['sample_data'])
    
    st.markdown("---")
    
    # Prediction Interface
    st.subheader("🔮 Make Predictions")
    st.write("Enter values for prediction:")
    
    model = st.session_state.trained_model
    columns_config = st.session_state.model_data['columns_config']
    
    # Create input fields based on model features
    input_data = {}
    
    for col, config in columns_config.items():
        if config['type'] == 'numeric':
            if 'min' in config and 'max' in config:
                input_data[col] = st.number_input(f"{col}", 
                                                min_value=float(config['min']), 
                                                max_value=float(config['max']), 
                                                value=float((config['min'] + config['max']) / 2))
            else:
                input_data[col] = st.number_input(f"{col}", value=0.0)
        else:
            input_data[col] = st.selectbox(f"{col}", config['values'])
    
    if st.button("🎯 Predict", use_container_width=True):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input data
            if model_type == 'healthcare':
                # Handle categorical encoding
                for col in input_df.select_dtypes(include=['object']).columns:
                    if col in model.label_encoders:
                        input_df[col] = model.label_encoders[col].transform(input_df[col])
                
                # Apply same preprocessing as training
                input_df = pd.DataFrame(model.imputer.transform(input_df), columns=input_df.columns)
                input_scaled = model.scaler.transform(input_df)
                
                # Make prediction
                prediction = model.model.predict(input_scaled)[0]
                probability = model.model.predict_proba(input_scaled)[0]
                
                st.success(f"**Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}**")
                st.info(f"Probability of Prediction bieng incorrect: {probability[1]:.4f}")
                
            else:  # cybersecurity
                # Apply same preprocessing as training
                input_scaled = model.scaler.transform(input_df)
                
                # Make prediction
                prediction = model.model.predict(input_scaled)[0]
                probability = model.model.predict_proba(input_scaled)[0]
                
                st.success(f"**Prediction: {'Attack Detected' if prediction == 1 else 'Benign Traffic'}**")
                st.info(f"Attack Probability: {probability[1]:.4f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Main app navigation
def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    if st.sidebar.button("🏠 Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.session_state.trained_model is not None:
        if st.sidebar.button("📈 Results & Prediction"):
            st.session_state.page = 'results'
            st.rerun()
    
    # Page routing
    if st.session_state.page == 'home':
        home_page()
    elif st.session_state.page == 'healthcare_config':
        healthcare_config_page()
    elif st.session_state.page == 'cybersecurity_config':
        cybersecurity_config_page()
    elif st.session_state.page == 'results':
        results_page()

if __name__ == "__main__":
    main()
