import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import plotly.express as px
from datetime import datetime

# Configuration
st.set_page_config(page_title="Network Security Dashboard", layout="wide")

# Data loading with caching
@st.cache_data
def load_data(samples=100):
    """Generate or load network traffic data"""
    try:
        np.random.seed(42)
        data = pd.DataFrame({
            'Timestamp': pd.date_range(start=datetime.now(), periods=samples, freq='T'),
            'IP_Address': np.random.choice(['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4'], samples),
            'Bytes_Sent': np.random.lognormal(mean=5, sigma=1, size=samples).astype(int),
            'Bytes_Received': np.random.lognormal(mean=5, sigma=1, size=samples).astype(int),
            'Request_Count': np.random.poisson(lam=25, size=samples),
            'Port': np.random.choice([80, 443, 22, 3389], samples)
        })
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Anomaly detection
def detect_anomalies(data, contamination=0.1):
    """Detect anomalies using Isolation Forest"""
    try:
        features = ['Bytes_Sent', 'Bytes_Received', 'Request_Count']
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Scale features for better model performance
        scaled_data = data[features].copy()
        anomaly_scores = model.fit_predict(scaled_data)
        data['Anomaly_Score'] = anomaly_scores
        data['Confidence'] = model.decision_function(scaled_data)
        return data
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        return data

# Visualization functions
def create_visualizations(data, anomalies):
    """Create interactive visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive scatter plot
        fig1 = px.scatter(
            data,
            x='Bytes_Sent',
            y='Bytes_Received',
            color='Anomaly_Score',
            size='Request_Count',
            hover_data=['IP_Address', 'Port', 'Timestamp'],
            title='Traffic Pattern Analysis'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Time series of anomalies
        fig2 = px.line(
            anomalies,
            x='Timestamp',
            y='Bytes_Sent',
            title='Anomalous Traffic Over Time',
            hover_data=['IP_Address', 'Port']
        )
        st.plotly_chart(fig2, use_container_width=True)

# Main dashboard
def main():
    st.title('Network Security Threat Detection Dashboard')
    
    # Sidebar controls
    st.sidebar.header('Settings')
    sample_size = st.sidebar.slider('Sample Size', 100, 1000, 100, 100)
    contamination = st.sidebar.slider('Contamination Level', 0.01, 0.5, 0.1, 0.01)
    refresh = st.sidebar.button('Refresh Data')
    
    # Load data
    data = load_data(sample_size)
    if data is None:
        return
    
    # Analyze data
    analyzed_data = detect_anomalies(data, contamination)
    anomalies = analyzed_data[analyzed_data['Anomaly_Score'] == -1]
    
    # Layout with tabs
    tab1, tab2, tab3 = st.tabs(['Overview', 'Raw Data', 'Anomalies'])
    
    with tab1:
        st.subheader('Dashboard Overview')
        col1, col2, col3 = st.columns(3)
        col1.metric('Total Packets', len(data))
        col2.metric('Anomalies Detected', len(anomalies))
        col3.metric('Anomaly Rate', f"{len(anomalies)/len(data)*100:.1f}%")
        
        create_visualizations(analyzed_data, anomalies)
    
    with tab2:
        st.subheader('Raw Network Traffic Data')
        st.dataframe(data.style.format({
            'Bytes_Sent': '{:,.0f}',
            'Bytes_Received': '{:,.0f}',
            'Timestamp': '{:%Y-%m-%d %H:%M:%S}'
        }))
    
    with tab3:
        st.subheader('Detailed Anomaly Report')
        st.dataframe(anomalies.style.highlight_max(
            subset=['Bytes_Sent', 'Bytes_Received', 'Request_Count'],
            color='red'
        ))
    
    # Status indicator
    st.sidebar.success("Monitoring Active")
    st.sidebar.write(f"Last Updated: {datetime.now():%H:%M:%S}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")