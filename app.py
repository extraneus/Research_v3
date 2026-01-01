import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="RCC & Hypertension Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_artifacts(disease):
    """Load models and hub probes for the selected disease"""
    try:
        if disease == "RCC":
            base = Path("models_rcc")
            suffix = "_rcc.pkl"
        else:
            base = Path("models_hypr")
            suffix = "_hypr.pkl"
        
        # Load hub probes
        hub_probes_path = base / "hub_probes.json"
        if not hub_probes_path.exists():
            st.error(f"Hub probes file not found: {hub_probes_path}")
            return None, None
            
        with open(hub_probes_path) as f:
            hub_probes = json.load(f)
        
        # Load models
        models = {}
        for pkl in base.glob(f"*{suffix}"):
            model_key = pkl.stem.replace(suffix.replace(".pkl", ""), "")
            models[model_key] = joblib.load(pkl)
        
        if not models:
            st.error(f"No models found in {base}")
            return None, None
            
        return models, hub_probes
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None, None

def validate_data(df, hub_probes):
    """Validate uploaded data"""
    issues = []
    
    # Check if dataframe is empty
    if df.empty:
        issues.append("Uploaded file is empty")
    
    # Check for required columns
    available_genes = set(df.columns)
    required_genes = set(hub_probes)
    missing_genes = required_genes - available_genes
    
    if missing_genes:
        issues.append(f"Missing {len(missing_genes)} hub genes (will be filled with 0)")
    
    return issues

def prepare_data(df, hub_probes):
    """Prepare data for prediction"""
    # Convert column names to string
    df.columns = df.columns.map(str)
    
    # Identify and store metadata columns
    metadata_cols = ["SampleID", "Sample_Description", "Label"]
    metadata = {}
    for col in metadata_cols:
        if col in df.columns:
            metadata[col] = df[col].copy()
    
    # Drop metadata columns
    df_genes = df.drop(columns=[c for c in metadata_cols if c in df.columns], errors='ignore')
    
    # Ensure all hub probes are present
    for gene in hub_probes:
        if gene not in df_genes.columns:
            df_genes[gene] = 0.0
    
    # Select only hub probes in correct order
    X = df_genes[hub_probes]
    
    return X, metadata

def create_probability_distribution(probabilities, threshold=0.5):
    """Create a distribution plot for probabilities"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=probabilities,
        nbinsx=30,
        name='Probability Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold})",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Prediction Probability Distribution",
        xaxis_title="Probability",
        yaxis_title="Frequency",
        showlegend=False,
        height=400
    )
    
    return fig

def create_prediction_pie(predictions):
    """Create a pie chart for prediction distribution"""
    pred_counts = pd.Series(predictions).value_counts()
    labels = ["Negative (0)", "Positive (1)"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=[pred_counts.get(0, 0), pred_counts.get(1, 0)],
        marker=dict(colors=['#2ecc71', '#e74c3c']),
        hole=0.3
    )])
    
    fig.update_layout(
        title="Prediction Distribution",
        height=400
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🧬 RCC & Hypertension Gene Expression Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Integrated Bioinformatics and Machine Learning for Cancer and Cardiovascular Disease Prediction</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/dna.png", width=80)
        st.title("Configuration")
        
        # Disease selection
        disease = st.selectbox(
            "🔬 Select Disease",
            ["RCC", "Hypertension"],
            help="Choose between Renal Cell Carcinoma or Hypertension prediction"
        )
        
        st.divider()
        
        # File upload
        st.subheader("📁 Upload Data")
        uploaded = st.file_uploader(
            "Upload Gene Expression CSV",
            type=["csv"],
            help="Upload a CSV file containing gene expression data"
        )
        
        st.divider()
        
        # Information
        with st.expander("ℹ️ About This Tool"):
            st.markdown("""
            This tool uses machine learning models trained on gene expression data to predict:
            
            **RCC (Renal Cell Carcinoma):**
            - Dataset: GSE68417 & GSE53000
            - Hub genes identified through network analysis
            
            **Hypertension:**
            - Dataset: GSE53408 & GSE113439
            - Hub genes from cardiovascular pathways
            
            **Methods:**
            - Differential gene expression analysis
            - Machine learning classification
            - Survival analysis integration
            """)
        
        with st.expander("📊 Data Format"):
            st.markdown("""
            **Required Format:**
            - CSV file with gene expression values
            - Rows: Samples
            - Columns: Gene probes/IDs
            
            **Optional Columns:**
            - SampleID
            - Sample_Description
            - Label (for validation)
            
            Missing hub genes will be filled with 0.
            """)
    
    # Load models and hub probes
    with st.spinner(f"Loading {disease} models and gene signatures..."):
        models, hub_probes = load_artifacts(disease)
    
    if models is None or hub_probes is None:
        st.error("Failed to load models. Please check your model files.")
        return
    
    # Display model information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Models", len(models))
    with col2:
        st.metric("Hub Genes", len(hub_probes))
    with col3:
        st.metric("Disease", disease)
    
    st.divider()
    
    # Model selection
    st.subheader("⚙️ Model Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_name = st.selectbox(
            "Select Prediction Model",
            list(models.keys()),
            help="Choose the machine learning model for prediction"
        )
    
    with col2:
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Samples with probability ≥ threshold will be classified as positive"
        )
    
    model = models[model_name]
    
    # Display hub genes
    with st.expander("🧬 View Hub Genes"):
        st.markdown(f"**Total Hub Genes:** {len(hub_probes)}")
        hub_df = pd.DataFrame({
            "Gene ID": hub_probes,
            "Index": range(1, len(hub_probes) + 1)
        })
        st.dataframe(hub_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Prediction section
    if uploaded:
        st.subheader("📊 Prediction Results")
        
        try:
            # Load data
            with st.spinner("Loading and validating data..."):
                df = pd.read_csv(uploaded)
                
                # Validate data
                issues = validate_data(df, hub_probes)
                if issues:
                    with st.expander("⚠️ Data Validation Warnings", expanded=True):
                        for issue in issues:
                            st.warning(issue)
                
                # Prepare data
                X, metadata = prepare_data(df, hub_probes)
                
                st.success(f"✅ Successfully loaded {len(X)} samples with {len(hub_probes)} hub genes")
            
            # Make predictions
            with st.spinner("Making predictions..."):
                # Get probabilities
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[:, 1]
                elif hasattr(model, "decision_function"):
                    # Normalize decision function to [0, 1]
                    scores = model.decision_function(X)
                    probs = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    st.error("Model does not support probability prediction")
                    return
                
                # Make predictions based on threshold
                preds = (probs >= threshold).astype(int)
                
                # Create results dataframe
                results = pd.DataFrame({
                    "Sample_Index": range(1, len(preds) + 1),
                    "Prediction": preds,
                    "Probability": probs,
                    "Classification": ["Positive" if p == 1 else "Negative" for p in preds]
                })
                
                # Add metadata if available
                for col, values in metadata.items():
                    results[col] = values.values
            
            # Display summary metrics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(results))
            with col2:
                positive_count = (results['Prediction'] == 1).sum()
                st.metric("Positive Cases", positive_count)
            with col3:
                negative_count = (results['Prediction'] == 0).sum()
                st.metric("Negative Cases", negative_count)
            with col4:
                avg_prob = results['Probability'].mean()
                st.metric("Avg. Probability", f"{avg_prob:.3f}")
            
            # Visualizations
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = create_probability_distribution(results['Probability'], threshold)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_pie = create_prediction_pie(results['Prediction'])
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Top predictions
            st.subheader("High-Risk Samples (Top 10 by Probability)")
            top_samples = results.nlargest(10, 'Probability')
            st.dataframe(
                top_samples.style.background_gradient(subset=['Probability'], cmap='Reds'),
                use_container_width=True,
                hide_index=True
            )
            
            # Full results table
            st.subheader("Complete Results")
            st.dataframe(results, use_container_width=True, hide_index=True)
            
            # Download options
            st.subheader("📥 Download Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = results.to_csv(index=False)
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv_data,
                    file_name=f"{disease.lower()}_predictions_{model_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                positive_only = results[results['Prediction'] == 1]
                if not positive_only.empty:
                    csv_positive = positive_only.to_csv(index=False)
                    st.download_button(
                        label="Download Positive Cases (CSV)",
                        data=csv_positive,
                        file_name=f"{disease.lower()}_positive_cases.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                high_risk = results[results['Probability'] >= 0.7]
                if not high_risk.empty:
                    csv_high_risk = high_risk.to_csv(index=False)
                    st.download_button(
                        label="Download High-Risk (≥0.7) (CSV)",
                        data=csv_high_risk,
                        file_name=f"{disease.lower()}_high_risk.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            # Model performance (if labels are available)
            if 'Label' in metadata:
                st.divider()
                st.subheader("📈 Model Performance")
                
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
                
                y_true = metadata['Label'].values
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, preds)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    if len(np.unique(y_true)) > 1:
                        auc = roc_auc_score(y_true, probs)
                        st.metric("AUC-ROC", f"{auc:.4f}")
                with col3:
                    sensitivity = (preds[y_true == 1] == 1).sum() / (y_true == 1).sum()
                    st.metric("Sensitivity", f"{sensitivity:.4f}")
                
                # Confusion matrix
                col1, col2 = st.columns(2)
                with col1:
                    cm = confusion_matrix(y_true, preds)
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Negative", "Positive"],
                        y=["Negative", "Positive"],
                        text_auto=True,
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    if len(np.unique(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, probs)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC={auc:.3f})',
                            line=dict(color='blue', width=2)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(color='red', dash='dash')
                        ))
                        fig_roc.update_layout(
                            title="ROC Curve",
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate"
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                # Classification report
                with st.expander("📋 Detailed Classification Report"):
                    report = classification_report(y_true, preds, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a gene expression CSV file to begin prediction")
        
        st.markdown("""
        ### 📖 How to Use This Tool
        
        1. **Select Disease**: Choose between RCC or Hypertension in the sidebar
        2. **Upload Data**: Upload your gene expression CSV file
        3. **Select Model**: Choose the machine learning model
        4. **Adjust Threshold**: Set the classification threshold (default: 0.5)
        5. **View Results**: Analyze predictions, visualizations, and download results
        
        ### 🔬 Research Background
        
        This tool implements machine learning models based on research integrating bioinformatics 
        and computational approaches to identify:
        
        - Common abnormal genes between RCC and Hypertension
        - Molecular pathways linking kidney cancer and cardiovascular disease
        - Hub genes with prognostic significance
        - Drug repurposing opportunities through DSigDB analysis
        
        **Key Findings:**
        - 102 Common Differentially Expressed Genes (DEGs)
        - 11 Hub genes with clinical significance
        - Survival analysis integration
        - Cross-disease molecular mechanisms
        """)

# Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><strong>Integrated Bioinformatics and Machine Learning for RCC & Hypertension</strong></p>
            <p>Datasets: GSE68417, GSE53000, GSE53408, GSE113439 | Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()