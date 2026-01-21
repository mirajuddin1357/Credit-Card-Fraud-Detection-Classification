# ---------------------------------------------------------
# CREDIT CARD FRAUD PROTECTION PLATFORM (AI CORE v1.0)
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import base64

# BASE DIRECTORY SETUP - ensuring file paths are relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- UTILITY: LOAD MODEL & DATA ---
@st.cache_resource # Cached for performance
def load_engine():
    try:
        model_path = os.path.join(BASE_DIR, 'fraud_detection_model.pkl')
        csv_path = os.path.join(BASE_DIR, 'creditcard.csv')
        
        if not os.path.exists(model_path):
            st.error(f"FATAL: Model file not found at {model_path}")
            return None, None, None
            
        # Load the serialized machine learning model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Initialize and fit dual scalers for Amount and Time to match training
        amount_scaler = StandardScaler()
        time_scaler = StandardScaler()
        
        if os.path.exists(csv_path):
            # Only read necessary columns for fitting
            df_temp = pd.read_csv(csv_path, usecols=['Amount', 'Time'])
            amount_scaler.fit(df_temp[['Amount']])
            time_scaler.fit(df_temp[['Time']])
            return model, amount_scaler, time_scaler
        else:
            st.error(f"FATAL: CSV file not found at {csv_path}")
            return None, None, None
            
    except Exception as e:
        st.error(f"Internal Engine Load Error: {e}")
        return None, None, None

@st.cache_data # Cached to avoid repeated disk reads
def load_data():
    try:
        csv_path = os.path.join(BASE_DIR, 'creditcard.csv')
        # Load the raw dataset
        df = pd.read_csv(csv_path)
        
        # Preprocessing: The app (and model) expects scaled features
        # We reuse a local scaler just for the dashboard visualization
        scaler = StandardScaler()
        
        # We preserve the original Amount for display but create Amount for the model
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return None

# --- GLOBAL STYLES (Futuristic Glassmorphism) ---
def apply_aesthetics():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root {
        --neon-red: #ff0800;
        --neon-blue: #00f2ff;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .stApp {
        background: radial-gradient(circle at top right, #1a0a0a, #000000);
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, var(--neon-red), var(--neon-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: var(--neon-red);
    }

    .stButton>button {
        background: linear-gradient(45deg, #ff0800, #ff4d4d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        transition: 0.3s all;
        box-shadow: 0 0 15px rgba(255, 8, 0, 0.4);
    }

    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(255, 8, 0, 0.8);
        transform: scale(1.05);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(26, 10, 10, 0.95);
        border-right: 1px solid var(--glass-border);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CORE PREDICTION ENGINE ---
def get_prediction(model, amount_scaler, time_scaler, raw_inputs):
    try:
        # Input comes in order: V1..V28 (indices 0-27), Time (28), Amount (29)
        v_features = raw_inputs[:28]
        time_raw = raw_inputs[28]
        amount_raw = raw_inputs[29]
        
        # Scale Amount and Time as done in training
        scaled_amount = amount_scaler.transform([[amount_raw]])[0][0]
        scaled_time = time_scaler.transform([[time_raw]])[0][0]
        
        # Model was trained with column order: Time, V1-V28, Amount
        # (Original CSV order: Time, V1..V28, Amount, Class)
        final_input = [scaled_time] + list(v_features) + [scaled_amount]
        
        fraud_prob = model.predict_proba([final_input])[0][1]
        return fraud_prob
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# --- MAIN APP ---
def main():
    apply_aesthetics()
    
    st.sidebar.markdown("<h2 style='text-align:center;'>Fraud Defense</h2>", unsafe_allow_html=True)
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Verify Transaction", "Risk Analytics", "About"])

    model, amount_scaler, time_scaler = load_engine()
    df = load_data()

    if df is None or model is None:
        st.warning("⚠️ PROTOTYPE MODE: Active protection limited.")
        if model is None:
            st.error("Model engine 'fraud_detection_model.pkl' not found.")
        if df is None:
            st.info("💡 Tip: Ensure 'creditcard.csv' is in the project directory.")

    # --- SIDEBAR: Security Tips - Enhanced user awareness ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Security Tips")
    tips = [
        "Never share your CVV or OTP with anyone.",
        "Monitor your bank statements weekly.",
        "Enable 2-factor authentication on all banking apps.",
        "Avoid using public Wi-Fi for financial transactions.",
        "Large transactions from unknown locations are high risk."
    ]
    # Rotate tips randomly on reload
    st.sidebar.info(np.random.choice(tips))

    # -----------------------------------------------------
    # PAGE: DASHBOARD
    # -----------------------------------------------------
    if menu == "Dashboard":
        st.markdown('<div class="glass-card"><h1>Fraud Dashboard</h1></div>', unsafe_allow_html=True)
        
        if df is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                df['Class_Label'] = df['Class'].map({0: 'Legitimate', 1: 'Fraud'})
                fig = px.pie(df, names='Class_Label', title='Fraud vs Legitimate', 
                             hole=0.4, color_discrete_sequence=['#00f2ff', '#ff0800'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                
                st.markdown('<div class="glass-card"><h3>Transactions by Class</h3>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                fig = px.violin(df,x="Class",y="Amount",color="Class",box=True,points="outliers",color_discrete_map={0: '#00f2ff', 1: '#ff0800'},title="Amount Distribution by Class",template="plotly_dark")

                fig.update_layout(xaxis_title="Class (0 = Legitimate, 1 = Fraud)",yaxis_title="Transaction Amount",paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')


                
                st.markdown('<div class="glass-card"><h3>Amount Distribution</h3>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📊 Dashboard data will appear here once the dataset is processed.")

    # -----------------------------------------------------
    # PAGE: VERIFY TRANSACTION
    # -----------------------------------------------------
    elif menu == "Verify Transaction":
        st.markdown('<div class="glass-card"><h1>Verify Transaction</h1>', unsafe_allow_html=True)
        
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        time = st.number_input("Time (Seconds from first transaction)", min_value=0.0, value=0.0)
        st.markdown("""<div style="text-align: center;"><h3>High-Impact Features</h3><p style="color: gray; font-size: 0.9em;">Adjust critical components discovered via Risk Analytics.</p></div>""", unsafe_allow_html=True)


        # -----------------------------------------------------
        # PROPORTIONAL FEATURES (Positive Correlation)
        # -----------------------------------------------------
        st.markdown("### Proportional Features")
        st.caption("Higher values increase fraud probability.")

        # Strong to Weak Positive
        v11 = st.number_input("V11 (Strong Positive)", value=float(df['V11'].mean()))
        v4 = st.number_input("V4 (Positive)", value=float(df['V4'].mean()))
        v2 = st.number_input("V2 (Positive)", value=float(df['V2'].mean()))
        v21 = st.number_input("V21 (Positive)", value=float(df['V21'].mean()))
        v19 = st.number_input("V19 (Positive)", value=float(df['V19'].mean()))
        v20 = st.number_input("V20 (Positive)", value=float(df['V20'].mean()))
        v8 = st.number_input("V8 (Positive)", value=float(df['V8'].mean()))
        v27 = st.number_input("V27 (Weak Positive)", value=float(df['V27'].mean()))
        v28 = st.number_input("V28 (Weak Positive)", value=float(df['V28'].mean()))

        # -----------------------------------------------------
        # INVERSE PROPORTIONAL FEATURES (Negative Correlation)
        # -----------------------------------------------------
        st.markdown("### Inverse Proportional Features")
        st.caption("Lower values increase fraud probability.")

        # Weak to Strong Negative
        v26 = st.number_input("V26 (Weak Negative)", value=float(df['V26'].mean()))
        v25 = st.number_input("V25 (Weak Negative)", value=float(df['V25'].mean()))
        v24 = st.number_input("V24 (Weak Negative)", value=float(df['V24'].mean()))
        v23 = st.number_input("V23 (Weak Negative)", value=float(df['V23'].mean()))
        v22 = st.number_input("V22 (Weak Negative)", value=float(df['V22'].mean()))
        v15 = st.number_input("V15 (Weak Negative)", value=float(df['V15'].mean()))
        v13 = st.number_input("V13 (Weak Negative)", value=float(df['V13'].mean()))
        v6 = st.number_input("V6 (Weak Negative)", value=float(df['V6'].mean()))
        v5 = st.number_input("V5 (Negative)", value=float(df['V5'].mean()))
        v9 = st.number_input("V9 (Negative)", value=float(df['V9'].mean()))
        v1 = st.number_input("V1 (Negative)", value=float(df['V1'].mean()))
        v18 = st.number_input("V18 (Negative)", value=float(df['V18'].mean()))
        v7 = st.number_input("V7 (Negative)", value=float(df['V7'].mean()))
        v3 = st.number_input("V3 (Negative)", value=float(df['V3'].mean()))
        v16 = st.number_input("V16 (Negative)", value=float(df['V16'].mean()))
        v10 = st.number_input("V10 (Negative)", value=float(df['V10'].mean()))
        v12 = st.number_input("V12 (Negative)", value=float(df['V12'].mean()))
        v14 = st.number_input("V14 (Negative)", value=float(df['V14'].mean()))
        v17 = st.number_input("V17 (Strong Negative)", value=float(df['V17'].mean()))

        if st.button("Analyze Risk"):
            if model is not None and amount_scaler is not None:
                # Raw Features: V1, V2, ... V28, time, amount
                # We map our sliders to their specific indices (V1=index 0, V2=index 1, etc.)
                # Indices: V10=9, V12=11, V14=13, V17=16, V4=3
                # Initializing with means ensures unused features don't skew the prediction
                input_data = [df[f'V{i}'].mean() for i in range(1, 29)] + [time, amount]
                # Maps all available sliders to their specific PCA indices (0-27)
                mapping = {
                    0:v1, 1:v2, 2:v3, 3:v4, 4:v5, 5:v6, 6:v7, 7:v8, 8:v9, 9:v10,
                    10:v11, 11:v12, 12:v13, 13:v14, 14:v15, 15:v16, 16:v17, 17:v18,
                    18:v19, 19:v20, 20:v21, 21:v22, 22:v23, 23:v24, 24:v25, 25:v26, 26:v27,
                    27:v28
                }
                for idx, val in mapping.items():
                    input_data[idx] = val

                # Uses the complete 'input_data' list containing all 30 features.
                prob = get_prediction(model, amount_scaler, time_scaler, input_data)
                
                if prob is not None:
                    risk_level = "HIGH" if prob > 0.5 else "LOW"
                    color = "#ff0800" if risk_level == "HIGH" else "#00f2ff"
                    
                    st.markdown(f"""
                    <div style="text-align:center; padding:20px; border-radius:15px; background:rgba(255, 8, 0, 0.1); border:1px solid {color};">
                        <h2 style='margin:0; font-size:1.2em; opacity:0.8;'>ANALYSIS RESULT</h2>
                        <h1 style='font-size:3.5em; margin:10px 0; color:{color};'>{risk_level} RISK</h1>
                        <p style='color:white; font-weight:bold; font-size:1.1em;'>
                            Fraud Probability: {prob:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    if risk_level == "HIGH":
                        st.error("🚨 Warning: This transaction shows patterns matching known fraud cases.")
                    else:
                        st.success("✅ Secure: This transaction appears legitimate.")
            else:
                st.error("Model engine is currently offline. Please run the training notebook.")

    # -----------------------------------------------------
    # PAGE: RISK ANALYTICS
    # -----------------------------------------------------
    elif menu == "Risk Analytics":
        st.markdown('<div class="glass-card" style="text-align: center;"><h3>Feature Correlation with Fraud (Target)</h3></div>', unsafe_allow_html=True)
        
        if df is not None:
            # Main correlation chart container
            st.markdown("""
                Correlation values indicate the strength and direction of the relationship between each feature 
                and the likelihood of fraud. 
                - **Positive Correlation:** As the feature value increases, the risk of fraud increases.
                - **Negative Correlation:** As the feature value increases, the risk of fraud decreases.
            """)

            # Calculate correlation
            corr = df.corr()['Class'].sort_values(ascending=False).drop('Class')
            
            # Create Plotly Bar Chart
            fig = px.bar(
                x=corr.index, 
                y=corr.values,
                labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
                title="Correlation Analysis (Relationship with Target)",
                color=corr.values,
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Highlight significant features with improved formatting
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="glass-card"><h3>📈 Positive Influencers</h3>', unsafe_allow_html=True)
                st.caption("Top features associated with HIGH fraud risk")
                top_pos = corr[corr > 0].head(5)
                for feature, val in top_pos.items():
                    st.markdown(f"**{feature}**: `{val:.4f}`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="glass-card"><h3>📉 Negative Influencers</h3>', unsafe_allow_html=True)
                st.caption("Top features associated with LOW fraud risk")
                top_neg = corr[corr < 0].tail(5).sort_values()
                for feature, val in top_neg.items():
                    st.markdown(f"**{feature}**: `{val:.4f}`")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card">
                <h3>💡 Strategic Insight</h3>
                <p>Features <b>V17, V14, V12,</b> and <b>V10</b> are critical. In most fraud datasets, their decrease 
                (negative correlation) is a high-confidence indicator of fraudulent activity.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Professional placeholder for empty state
            st.markdown("""
            <div class="glass-card">
                <h3>🔬 Engine Readiness Required</h3>
                <p>The Risk Analytics engine is currently in standby. To activate this module:</p>
                <ul>
                    <li>Upload or ensure <b>creditcard.csv</b> is present.</li>
                    <li>The system will automatically compute feature-target relationships upon loading.</li>
                </ul>
                <p style="color:var(--neon-blue);"><i>Expected Analysis: Deep correlation profiling of V1-V28 PCA components.</i></p>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # PAGE: ABOUT
    # -----------------------------------------------------
    elif menu == "About":
        st.title("About Fraud Defense AI")
        st.markdown("""
        <div class="glass-card">
            <h3>Project Vision</h3>
            <p>Our mission is to safeguard digital transactions using state-of-the-art Deep Learning and Ensemble Machine Learning.</p>
            <p>Built by <b>Miraj Ud Din</b> as part of the Advanced ML Portfolio Series.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
