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
import uuid

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
if 'custom_columns' not in st.session_state:
    st.session_state.custom_columns = {}

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
                if 'distribution' in config:
                    if config['distribution'] == 'uniform':
                        data[col] = np.random.uniform(config['min'], config['max'], n_samples)
                    elif config['distribution'] == 'normal':
                        mean = (config['min'] + config['max']) / 2
                        std = (config['max'] - config['min']) / 6
                        data[col] = np.random.normal(mean, std, n_samples)
                        data[col] = np.clip(data[col], config['min'], config['max'])
                    elif config['distribution'] == 'exponential':
                        scale = config.get('scale', 1.0)
                        data[col] = np.random.exponential(scale, n_samples)
                else:
                    # Default uniform distribution
                    data[col] = np.random.uniform(config['min'], config['max'], n_samples)
            elif config['type'] == 'categorical':
                data[col] = np.random.choice(config['values'], n_samples)
            elif config['type'] == 'boolean':
                data[col] = np.random.choice([True, False], n_samples)
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk configuration
        risk_prob = np.full(n_samples, 0.1)  # Base risk
        
        for col, weight in risk_config.items():
            if col in df.columns and weight > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # For numeric columns, use quantile-based risk
                    risk_prob += weight * (df[col] > df[col].quantile(0.7))
                elif df[col].dtype == 'bool':
                    # For boolean columns
                    risk_prob += weight * df[col]
                else:
                    # For categorical columns, assign risk to specific categories
                    high_risk_categories = ['Abnormal', 'Emergency', 'Diabetes', 'Hypertension', 'Critical']
                    risk_prob += weight * df[col].isin(high_risk_categories)
        
        # Create probabilistic target
        df['High_Risk'] = np.random.binomial(1, np.clip(risk_prob, 0, 1), len(df))
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        X = df.drop(['High_Risk'], axis=1)
        y = df['High_Risk']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
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
                if 'distribution' in config:
                    if config['distribution'] == 'exponential':
                        scale = config.get('scale', 1000)
                        data[col] = np.random.exponential(scale, n_samples)
                    elif config['distribution'] == 'poisson':
                        lambda_val = config.get('lambda', 50)
                        data[col] = np.random.poisson(lambda_val, n_samples)
                    elif config['distribution'] == 'normal':
                        mean = config.get('mean', 0)
                        std = config.get('std', 1)
                        data[col] = np.random.normal(mean, std, n_samples)
                    else:
                        # Uniform distribution
                        data[col] = np.random.uniform(config['min'], config['max'], n_samples)
                else:
                    # Default normal distribution
                    mean = config.get('mean', 0)
                    std = config.get('std', 1)
                    data[col] = np.random.normal(mean, std, n_samples)
            elif config['type'] == 'categorical':
                data[col] = np.random.choice(config['values'], n_samples)
            elif config['type'] == 'boolean':
                data[col] = np.random.choice([True, False], n_samples)
        
        df = pd.DataFrame(data)
        
        # Create attack probability based on risk configuration
        attack_prob = np.full(n_samples, 0.05)  # Base attack probability
        
        for col, weight in risk_config.items():
            if col in df.columns and weight > 0:
                if df[col].dtype in ['int64', 'float64']:
                    attack_prob += weight * (df[col] > df[col].quantile(0.9))
                elif df[col].dtype == 'bool':
                    attack_prob += weight * df[col]
                else:
                    # For categorical columns
                    attack_categories = ['Attack', 'Malicious', 'Suspicious']
                    attack_prob += weight * df[col].isin(attack_categories)
        
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
        
        # Handle categorical and boolean variables
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
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

def add_custom_column():
    """Add a new custom column"""
    new_id = str(uuid.uuid4())
    st.session_state.custom_columns[new_id] = {
        'name': f'Custom_Column_{len(st.session_state.custom_columns) + 1}',
        'type': 'numeric',
        'config': {'min': 0, 'max': 100}
    }

def delete_custom_column(column_id):
    """Delete a custom column"""
    if column_id in st.session_state.custom_columns:
        del st.session_state.custom_columns[column_id]

def render_column_configuration(model_type):
    """Render dynamic column configuration interface"""
    st.subheader("🔧 Column Configuration")
    
    # Add new column button
    if st.button("➕ Add Custom Column", use_container_width=True):
        add_custom_column()
        st.rerun()
    
    st.markdown("---")
    
    # Default columns based on model type
    if model_type == 'healthcare':
        default_columns = {
            'default_age': {
                'name': 'Age',
                'type': 'numeric',
                'config': {'min': 18, 'max': 90, 'distribution': 'uniform'}
            },
            'default_gender': {
                'name': 'Gender',
                'type': 'categorical',
                'config': {'values': ['Male', 'Female']}
            },
            'default_blood_type': {
                'name': 'Blood_Type',
                'type': 'categorical',
                'config': {'values': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']}
            },
            'default_medical_condition': {
                'name': 'Medical_Condition',
                'type': 'categorical',
                'config': {'values': ['Diabetes', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis']}
            },
            'default_admission_type': {
                'name': 'Admission_Type',
                'type': 'categorical',
                'config': {'values': ['Emergency', 'Elective', 'Urgent']}
            },
            'default_insurance_provider': {
                'name': 'Insurance_Provider',
                'type': 'categorical',
                'config': {'values': ['Aetna', 'Blue Cross', 'Cigna', 'Medicare', 'UnitedHealthcare']}
            },
            'default_billing_amount': {
                'name': 'Billing_Amount',
                'type': 'numeric',
                'config': {'min': 1000, 'max': 50000, 'distribution': 'uniform'}
            },
            'default_days_in_hospital': {
                'name': 'Days_in_Hospital',
                'type': 'numeric',
                'config': {'min': 1, 'max': 30, 'distribution': 'uniform'}
            },
            'default_test_results': {
                'name': 'Test_Results',
                'type': 'categorical',
                'config': {'values': ['Normal', 'Abnormal', 'Inconclusive']}
            }
        }
    else:  # cybersecurity
        default_columns = {
            'default_flow_duration': {
                'name': 'Flow_Duration',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 1000}
            },
            'default_total_fwd_packets': {
                'name': 'Total_Fwd_Packets',
                'type': 'numeric',
                'config': {'distribution': 'poisson', 'lambda': 50}
            },
            'default_total_backward_packets': {
                'name': 'Total_Backward_Packets',
                'type': 'numeric',
                'config': {'distribution': 'poisson', 'lambda': 30}
            },
            'default_total_length_fwd_packets': {
                'name': 'Total_Length_of_Fwd_Packets',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 2000}
            },
            'default_total_length_bwd_packets': {
                'name': 'Total_Length_of_Bwd_Packets',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 1500}
            },
            'default_flow_bytes_per_second': {
                'name': 'Flow_Bytes_per_Second',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 5000}
            },
            'default_flow_packets_per_second': {
                'name': 'Flow_Packets_per_Second',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 10}
            },
            'default_flow_iat_mean': {
                'name': 'Flow_IAT_Mean',
                'type': 'numeric',
                'config': {'distribution': 'exponential', 'scale': 100}
            },
            'default_psh_flag_count': {
                'name': 'PSH_Flag_Count',
                'type': 'numeric',
                'config': {'distribution': 'poisson', 'lambda': 2}
            },
            'default_urg_flag_count': {
                'name': 'URG_Flag_Count',
                'type': 'numeric',
                'config': {'distribution': 'poisson', 'lambda': 0.1}
            },
            'default_average_packet_size': {
                'name': 'Average_Packet_Size',
                'type': 'numeric',
                'config': {'distribution': 'normal', 'mean': 150, 'std': 30}
            }
        }

    # Combine default and custom columns
    all_columns = {**default_columns, **st.session_state.custom_columns}
    
    columns_config = {}
    
    # Render each column configuration
    for col_id, col_data in all_columns.items():
        with st.expander(f"Configure: {col_data['name']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Column name
                col_name = st.text_input(
                    "Column Name", 
                    value=col_data['name'], 
                    key=f"name_{col_id}"
                )
                
                # Column type
                col_type = st.selectbox(
                    "Column Type",
                    options=['numeric', 'categorical', 'boolean'],
                    index=['numeric', 'categorical', 'boolean'].index(col_data['type']),
                    key=f"type_{col_id}"
                )
                
                # Type-specific configuration
                if col_type == 'numeric':
                    distribution = st.selectbox(
                        "Distribution",
                        options=['uniform', 'normal', 'exponential', 'poisson'],
                        key=f"dist_{col_id}"
                    )
                    
                    if distribution in ['uniform', 'normal']:
                        min_val = st.number_input(
                            "Min Value", 
                            value=float(col_data['config'].get('min', 0)),
                            key=f"min_{col_id}"
                        )
                        max_val = st.number_input(
                            "Max Value", 
                            value=float(col_data['config'].get('max', 100)),
                            key=f"max_{col_id}"
                        )
                        config = {'min': min_val, 'max': max_val, 'distribution': distribution}
                    elif distribution == 'exponential':
                        scale = st.number_input(
                            "Scale Parameter", 
                            value=float(col_data['config'].get('scale', 1.0)),
                            min_value=0.1,
                            key=f"scale_{col_id}"
                        )
                        config = {'distribution': distribution, 'scale': scale}
                    elif distribution == 'poisson':
                        lambda_val = st.number_input(
                            "Lambda Parameter", 
                            value=float(col_data['config'].get('lambda', 5.0)),
                            min_value=0.1,
                            key=f"lambda_{col_id}"
                        )
                        config = {'distribution': distribution, 'lambda': lambda_val}
                    
                elif col_type == 'categorical':
                    values_text = st.text_area(
                        "Categories (comma-separated)",
                        value=', '.join(col_data['config'].get('values', ['Option1', 'Option2'])),
                        key=f"values_{col_id}"
                    )
                    config = {'values': [v.strip() for v in values_text.split(',') if v.strip()]}
                    
                else:  # boolean
                    config = {}
                
                # Include in model
                include_col = st.checkbox(
                    "Include in Model", 
                    value=True, 
                    key=f"include_{col_id}"
                )
                
                if include_col:
                    columns_config[col_name] = {'type': col_type, **config}
            
            with col2:
                # Delete button (only for custom columns)
                if col_id.startswith('default_'):
                    st.write("*Default Column*")
                else:
                    if st.button("🗑️ Delete", key=f"delete_{col_id}"):
                        delete_custom_column(col_id)
                        st.rerun()
    
    return columns_config

def render_risk_configuration(columns_config):
    """Render risk parameter configuration"""
    st.subheader("⚠️ Risk Parameters Configuration")
    st.write("Set the risk probability weights for each column (0.0 to 1.0)")
    
    risk_config = {}
    
    # Create columns for better layout
    cols = st.columns(2)
    col_names = list(columns_config.keys())
    
    for i, col_name in enumerate(col_names):
        with cols[i % 2]:
            weight = st.slider(
                f"Risk weight for {col_name}",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key=f"risk_{col_name}",
                help=f"Higher values mean {col_name} has more influence on risk prediction"
            )
            risk_config[col_name] = weight
    
    return risk_config

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
        st.write("Predict high-risk patients based on customizable medical data features.")
        if st.button("🏥 Healthcare Model", key="healthcare", use_container_width=True):
            st.session_state.page = 'healthcare_config'
            st.session_state.model_type = 'healthcare'
            st.session_state.custom_columns = {}  # Reset custom columns
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🔒 Cybersecurity Threat Prediction")
        st.write("Detect network attacks and threats based on customizable network features.")
        if st.button("🔒 Cybersecurity Model", key="cybersecurity", use_container_width=True):
            st.session_state.page = 'cybersecurity_config'
            st.session_state.model_type = 'cybersecurity'
            st.session_state.custom_columns = {}  # Reset custom columns
            st.rerun()

def healthcare_config_page():
    """Configuration page for healthcare model"""
    st.title("🏥 Healthcare Model Configuration")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Configuration")
    
    # Dataset size
    n_samples = st.number_input(
        "Number of samples to generate", 
        min_value=1000, 
        max_value=500000, 
        value=50000, 
        step=1000
    )
    
    # Column configuration
    columns_config = render_column_configuration('healthcare')
    
    if not columns_config:
        st.warning("Please configure at least one column to proceed.")
        return
    
    # Risk configuration
    risk_config = render_risk_configuration(columns_config)
    
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
                'risk_config': risk_config,
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
    
    st.subheader("📊 Dataset Configuration")
    
    # Dataset size
    n_samples = st.number_input(
        "Number of samples to generate", 
        min_value=1000, 
        max_value=500000, 
        value=50000, 
        step=1000
    )
    
    # Column configuration
    columns_config = render_column_configuration('cybersecurity')
    
    if not columns_config:
        st.warning("Please configure at least one column to proceed.")
        return
    
    # Risk configuration
    risk_config = render_risk_configuration(columns_config)
    
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
                'risk_config': risk_config,
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
    columns_config = st.session_state.model_data['columns_config']
    risk_config = st.session_state.model_data['risk_config']
    
    st.markdown("---")
    
    # Display results
    st.subheader("🎯 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.4f}")
    
    with col2:
        st.metric("AUC Score", f"{results['auc_score']:.4f}")
    
    with col3:
        st.metric("Model Type", model_type.title())
    
    with col4:
        st.metric("Features Used", len(columns_config))
    
    # Risk Configuration Summary
    with st.expander("📋 Risk Configuration Summary"):
        risk_df = pd.DataFrame([
            {'Column': col, 'Risk Weight': weight} 
            for col, weight in risk_config.items()
        ])
        st.dataframe(risk_df, use_container_width=True)
    
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
    importance_df = pd.DataFrame(list(results['feature_importance'].items()), columns=['Feature', 'Importance'])
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
    
    # Create input fields based on model features
    input_data = {}
    
    cols = st.columns(2)
    col_names = list(columns_config.keys())
    
    for i, (col, config) in enumerate(columns_config.items()):
        with cols[i % 2]:
            if config['type'] == 'numeric':
                if 'min' in config and 'max' in config:
                    input_data[col] = st.number_input(
                        f"{col}", 
                        min_value=float(config['min']), 
                        max_value=float(config['max']), 
                        value=float((config['min'] + config['max']) / 2),
                        key=f"pred_{col}"
                    )
                else:
                    input_data[col] = st.number_input(f"{col}", value=0.0, key=f"pred_{col}")
            elif config['type'] == 'categorical':
                input_data[col] = st.selectbox(f"{col}", config['values'], key=f"pred_{col}")
            else:  # boolean
                input_data[col] = st.checkbox(f"{col}", key=f"pred_{col}")
    
    if st.button("🎯 Predict", use_container_width=True):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input data
            if model_type == 'healthcare':
                # Handle categorical and boolean encoding
                for col in input_df.columns:
                    if col in model.label_encoders:
                        input_df[col] = model.label_encoders[col].transform(input_df[col].astype(str))
                    elif input_df[col].dtype == 'bool':
                        input_df[col] = input_df[col].astype(int)
                
                # Apply same preprocessing as training
                input_df = pd.DataFrame(model.imputer.transform(input_df), columns=input_df.columns)
                input_scaled = model.scaler.transform(input_df)
                
                # Make prediction
                prediction = model.model.predict(input_scaled)[0]
                probability = model.model.predict_proba(input_scaled)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}**")
                with col2:
                    st.info(f"Risk Probability: {probability[1]:.4f}")
                
            else:  # cybersecurity
                # Handle categorical and boolean encoding
                for col in input_df.columns:
                    if input_df[col].dtype in ['object', 'bool']:
                        le = LabelEncoder()
                        # Fit with dummy data to avoid issues
                        le.fit(['dummy', str(input_df[col].iloc[0])])
                        input_df[col] = le.transform([str(input_df[col].iloc[0])])
                
                # Apply same preprocessing as training
                input_scaled = model.scaler.transform(input_df)
                
                # Make prediction
                prediction = model.model.predict(input_scaled)[0]
                probability = model.model.predict_proba(input_scaled)[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Prediction: {'Attack Detected' if prediction == 1 else 'Benign Traffic'}**")
                with col2:
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
    
    # Display current configuration info in sidebar
    if st.session_state.custom_columns:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Custom Columns:**")
        for col_data in st.session_state.custom_columns.values():
            st.sidebar.write(f"• {col_data['name']} ({col_data['type']})")
    
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
