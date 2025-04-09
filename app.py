import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# HELPER FUNCTIONS (IMPROVED)
# --------------------------------------------------------------

def infer_columns(df: pd.DataFrame) -> dict:
    """Enhanced column detection with regex patterns and priority scoring"""
    patterns = {
        'timestamp': r'(timestamp|time|date|dt)$',
        'amount': r'(amount|amt|value|transaction_amt)$',
        'merchant': r'(merchant|vendor|retailer|business)',
        'cardholder': r'(cardholder|customer|client|accountholder|user)',
        'label': r'(class|label|is_fraud|fraud_flag|target)'
    }
    
    mappings = {}
    for col in df.columns:
        col_lower = col.lower()
        for key, pattern in patterns.items():
            if re.search(pattern, col_lower) and key not in mappings:
                mappings[key] = col
                break  # First match wins
    return mappings

# --------------------------------------------------------------
# DATA HANDLING (IMPROVED WITH CACHING)
# --------------------------------------------------------------

@st.cache_data
def load_csv_data(uploaded_file):
    """Load and cache CSV data with improved error handling"""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

@st.cache_data
def simulate_synthetic_data(n_samples=10000, fraud_ratio=0.01, random_state=42):
    """Enhanced synthetic data with more realistic patterns"""
    np.random.seed(random_state)
    df = pd.DataFrame()
    
    # Time features
    start_date = datetime(2020, 1, 1)
    df['Timestamp'] = [start_date + timedelta(minutes=int(x)) for x in np.random.randint(0, 525600, n_samples)]
    
    # Transaction features
    df['Amount'] = np.round(np.random.exponential(scale=50, size=n_samples), 2) + 0.01
    df['IsForeign'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    df['Category'] = np.random.choice(['Retail', 'Travel', 'Food', 'Services'], size=n_samples, p=[0.6, 0.1, 0.2, 0.1])
    
    # Fraud patterns
    n_fraud = int(n_samples * fraud_ratio)
    df['Class'] = 0
    fraud_indices = np.random.choice(df.index, size=n_fraud, replace=False)
    
    # Fraud characteristics
    df.loc[fraud_indices, 'Amount'] *= np.random.uniform(2, 5, size=n_fraud)
    df.loc[fraud_indices, 'IsForeign'] = np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7])
    df.loc[fraud_indices, 'Category'] = np.random.choice(['Travel', 'Services'], size=n_fraud, p=[0.7, 0.3])
    df.loc[fraud_indices, 'Class'] = 1
    
    return df

# --------------------------------------------------------------
# PREPROCESSING PIPELINE (IMPROVED)
# --------------------------------------------------------------

def build_preprocessor(df: pd.DataFrame, mappings: dict) -> Pipeline:
    """Create a reusable preprocessing pipeline"""
    # Identify feature types
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove special columns from features
    for col in mappings.values():
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# --------------------------------------------------------------
# MODELS (IMPROVED WITH HYPERPARAMETERS)
# --------------------------------------------------------------

def run_isolation_forest(X, contamination=0.01, n_estimators=100, max_samples='auto'):
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42
    )
    model.fit(X)
    scores = model.decision_function(X)
    preds = model.predict(X)
    return np.where(preds == -1, 1, 0), scores

def build_autoencoder(input_dim: int, dropout_rate=0.2):
    """Improved autoencoder architecture with dropout"""
    inputs = Input(shape=(input_dim,))
    x = Dense(int(input_dim * 0.8), activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(int(input_dim * 0.4), activation='relu')(x)
    encoded = Dropout(dropout_rate)(x)
    
    x = Dense(int(input_dim * 0.4), activation='relu')(encoded)
    x = Dropout(dropout_rate)(x)
    x = Dense(int(input_dim * 0.8), activation='relu')(x)
    decoded = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# --------------------------------------------------------------
# VISUALIZATIONS (IMPROVED WITH PLOTLY)
# --------------------------------------------------------------

def plot_anomaly_scores(scores: np.ndarray, algorithm_name: str):
    """Interactive histogram with Plotly"""
    fig = px.histogram(
        x=scores, 
        nbins=50,
        title=f"{algorithm_name} - Anomaly Score Distribution",
        labels={'x': 'Anomaly Score', 'y': 'Count'}
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """Precision-Recall curve visualization"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='PR Curve',
        line=dict(color='royalblue', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=700,
        height=500
    )
    st.plotly_chart(fig)

# --------------------------------------------------------------
# STREAMLIT APP (IMPROVED UI)
# --------------------------------------------------------------

def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        data_source = st.radio("Data Source", ["Upload CSV", "Generate Synthetic"])
        
        if data_source == "Generate Synthetic":
            n_samples = st.number_input("Sample Size", 1000, 100000, 10000)
            fraud_ratio = st.slider("Fraud Ratio", 0.1, 5.0, 1.0) / 100
            
        algorithm = st.selectbox(
            "Algorithm",
            ["Isolation Forest", "One-Class SVM", "LOF", "Autoencoder"]
        )
        
        contamination = st.slider(
            "Contamination Rate", 
            0.001, 0.1, 0.01, 
            help="Expected proportion of anomalies in the dataset"
        )
    
    # Main Content
    st.title("Credit Card Fraud Detection Analytics")
    st.write("Real-time anomaly detection and transaction monitoring")
    
    # Data Loading
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload transaction data", type=["csv"])
        if not uploaded_file:
            st.warning("Please upload a CSV file or switch to synthetic data")
            return
        df = load_csv_data(uploaded_file)
    else:
        df = simulate_synthetic_data(n_samples, fraud_ratio)
    
    if df is None:
        return
    
    # Data Exploration
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Dataset Shape: {df.shape}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            if 'Class' in df.columns:
                st.metric("Fraud Rate", f"{df['Class'].mean()*100:.2f}%")
    
    # Preprocessing
    with st.spinner("Preprocessing data..."):
        mappings = infer_columns(df)
        preprocessor = build_preprocessor(df, mappings)
        
        # Handle label separately
        label_col = mappings.get('label', 'Class') if 'label' in mappings else None
        y = df[label_col] if label_col and label_col in df else None
        
        # Transform features
        X = preprocessor.fit_transform(df)
        feature_names = (
            preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
            preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
        )
        X = pd.DataFrame(X, columns=feature_names)
    
    # Model Training
    st.header("Anomaly Detection Analysis")
    
    if algorithm == "Autoencoder":
        with st.expander("Autoencoder Configuration"):
            epochs = st.number_input("Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [32, 64, 128], index=0)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
    
    if st.button("Run Analysis"):
        with st.spinner(f"Running {algorithm}..."):
            if algorithm == "Isolation Forest":
                preds, scores = run_isolation_forest(
                    X, 
                    contamination=contamination,
                    n_estimators=200,
                    max_samples=0.8
                )
                
            elif algorithm == "One-Class SVM":
                model = OneClassSVM(nu=contamination, kernel='rbf')
                model.fit(X)
                scores = model.decision_function(X)
                preds = np.where(model.predict(X) == -1, 1, 0)
                
            elif algorithm == "LOF":
                model = LocalOutlierFactor(
                    n_neighbors=20, 
                    contamination=contamination, 
                    novelty=True
                )
                model.fit(X)
                scores = model.negative_outlier_factor_
                preds = np.where(model.predict(X) == -1, 1, 0)
                
            elif algorithm == "Autoencoder":
                model = build_autoencoder(X.shape[1], dropout_rate)
                early_stop = EarlyStopping(patience=5, restore_best_weights=True)
                
                history = model.fit(
                    X, X,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Plot training history
                fig = px.line(
                    history.history,
                    y=['loss', 'val_loss'],
                    labels={'index': 'Epoch', 'value': 'Loss'},
                    title='Training History'
                )
                st.plotly_chart(fig)
                
                preds = model.predict(X)
                mse = np.mean(np.power(X - preds, 2), axis=1)
                threshold = np.percentile(mse, 100*(1-contamination))
                preds = (mse > threshold).astype(int)
                scores = mse
        
        # Results Visualization
        col1, col2 = st.columns(2)
        with col1:
            plot_anomaly_scores(scores, algorithm)
            
        with col2:
            if y is not None:
                plot_pr_curve(y, scores)
        
        # Performance Metrics
        if y is not None:
            st.subheader("Performance Metrics")
            st.write(classification_report(y, preds, target_names=['Normal', 'Fraud']))
            
            cm = confusion_matrix(y, preds)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Normal', 'Fraud'],
                y=['Normal', 'Fraud'],
                text_auto=True
            )
            st.plotly_chart(fig)
            
if __name__ == "__main__":
    main()