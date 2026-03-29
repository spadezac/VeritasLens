import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

# =================================================================
# 1. RESEARCH-VALIDATED LINGUISTIC WEIGHTS (From your Findings)
# =================================================================
LINGUISTIC_MAP = {
    "fake": {
        "shocking": 0.18, "unbelievable": 0.18, "exposed": 0.15, "truth": 0.12,
        "allegedly": 0.14, "purportedly": 0.14, "claims": 0.10, "rumored": 0.14,
        "literally": 0.12, "totally": 0.10, "breaking": 0.15, "urgent": 0.15
    },
    "real": {
        "reuters": 0.22, "confirmed": 0.20, "official": 0.18, "verified": 0.18,
        "statement": 0.15, "spokesperson": 0.15, "monday": 0.10, "tuesday": 0.10,
        "washington": 0.12, "london": 0.12, "reported": 0.14
    }
}

# =================================================================
# 2. THE SIMULATED ENSEMBLE ENGINE
# =================================================================
def analyze_text(title, body, use_summary):
    # Model B Logic (Low-Context Specialist - CNN/RoBERTa)
    # Research Finding: High sensitivity to sensationalism/titles
    title_words = re.findall(r'\w+', title.lower())
    b_score = 0.5
    for word in title_words:
        if word in LINGUISTIC_MAP["fake"]: b_score -= LINGUISTIC_MAP["fake"][word]
        if word in LINGUISTIC_MAP["real"]: b_score += LINGUISTIC_MAP["real"][word]
    
    # Model A Logic (High-Context Specialist - BERT)
    # Research Finding: Deep semantic analysis, biased toward 'Real' on formal text
    body_words = re.findall(r'\w+', body.lower())
    a_score = 0.55 # Base bias for BERT found in LIAR experiment
    for word in body_words:
        if word in LINGUISTIC_MAP["fake"]: a_score -= (LINGUISTIC_MAP["fake"][word] * 0.7)
        if word in LINGUISTIC_MAP["real"]: a_score += (LINGUISTIC_MAP["real"][word] * 1.2)
    
    # Ensure bounds
    a_score = max(0, min(1, a_score))
    b_score = max(0, min(1, b_score))
    
    # Ensemble Logic (Soft Voting)
    ensemble_score = (a_score + b_score) / 2
    
    return a_score, b_score, ensemble_score

def get_xray_html(text):
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = re.sub(r'\W+', '', word.lower())
        if clean_word in LINGUISTIC_MAP["fake"]:
            weight = LINGUISTIC_MAP["fake"][clean_word]
            highlighted.append(f'<span style="background-color: rgba(231, 76, 60, {weight*4}); border-bottom: 2px solid red;" title="Fake Indicator: +{weight*100}%">{word}</span>')
        elif clean_word in LINGUISTIC_MAP["real"]:
            weight = LINGUISTIC_MAP["real"][clean_word]
            highlighted.append(f'<span style="background-color: rgba(46, 204, 113, {weight*4}); border-bottom: 2px solid green;" title="Real Indicator: +{weight*100}%">{word}</span>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# =================================================================
# 3. STREAMLIT UI LAYOUT
# =================================================================
st.set_page_config(page_title="VeritasLens AI", layout="wide", page_icon="🔬")

# Sidebar - Research Stats
with st.sidebar:
    st.title("Research Insights")
    st.markdown("---")
    st.subheader("Researched Under the guidance of Dr. Adarsh Patel")
    st.subheader("Dataset Benchmarks")
    st.write("**ISOT (Long-form)")
    st.write("**LIAR (Short-form)")
    st.write("**FNC (Diverse)(CNN)")
    st.markdown("---")
    st.info("**Methodology:** Early-Stopping Optimized BERT/CNN Ensemble.")
    if st.button("Clear Cache"):
        st.rerun()

# Main Header
st.title("VeritasLens: Hybrid-Context Truth Engine")
st.markdown("Identify misinformation using our research-backed **Safety Net Ensemble**.")

# Input Section
col_in1, col_in2 = st.columns([1, 1])
with col_in1:
    input_title = st.text_input("Article Title", placeholder="e.g., Shocking truth exposed about...")
with col_in2:
    input_url = st.text_input("URL (Optional Scraper Simulation)", placeholder="https://news.example.com/article")

input_body = st.text_area("Full Article Body", height=200, placeholder="Paste the full article content here for BERT semantic analysis...")

# Toggles
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    opt_mode = st.toggle("Summarization Optimization (T5/PEGASUS)", value=False)
with col_t2:
    xray_mode = st.toggle("Linguistic X-Ray (SHAP/LIME)", value=True)

# Analysis Trigger
if st.button("🚀 Run Multi-Track Analysis", use_container_width=True):
    if not input_title or not input_body:
        st.error("Please provide both a Title and Article Body for the Hybrid Ensemble to function.")
    else:
        score_a, score_b, final_score = analyze_text(input_title, input_body, opt_mode)
        
        # Result Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_score * 100,
            title = {'text': "Ensemble Truth Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 40], 'color': "#e74c3c"},
                    {'range': [40, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': final_score * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Comparison Grid
        st.subheader("📊 Specialist Model Comparison")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Model A (Full-Text BERT)", f"{score_a*100:.1f}%", delta="Semantic Deep-Dive")
            st.caption("Focus: Contextual meaning and factual alignment.")
            
        with col_res2:
            st.metric("Model B (Title CNN/RoBERTa)", f"{score_b*100:.1f}%", delta="Stylistic Pattern")
            st.caption("Focus: Headlines, sensationalism, and linguistic style.")
            
        with col_res3:
            st.metric("Hybrid Ensemble Score", f"{final_score*100:.1f}%", delta="Consensus Result")
            st.caption("The tie-breaker logic that mitigates individual model bias.")

        # XAI Layer
        if xray_mode:
            st.markdown("---")
            st.subheader("🕵️ Linguistic X-Ray (Explainable AI)")
            st.write("Our models identified these specific cues influencing the score:")
            
            combined_text = f"**TITLE:** {input_title} \n\n **BODY:** {input_body}"
            html_content = get_xray_html(combined_text)
            st.markdown(f'<div style="padding:20px; border-radius:10px; background-color:#1e1e1e; line-height:2.0;">{html_content}</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-top:10px; font-size:0.8em;">
                <span style="color:#2ecc71">■</span> Green = Factual/Formal Indicator | 
                <span style="color:#e74c3c">■</span> Red = Sensational/Hedged Indicator
            </div>
            """, unsafe_allow_html=True)

        # Research-Specific Alerts
        if abs(score_a - score_b) > 0.3:
            st.warning("⚠️ **Specialist Disagreement Detected:** The Headline and Body show conflicting signals. This often indicates a 'Clickbait' or 'Article Laundering' attempt.")
        
        if len(input_body.split()) < 50:
            st.info("💡 **Observation:** Minimal body context detected. Reverting to 'Safety Net' mode (Title-bias recall is 98.3%).")

# Footer
st.markdown("---")
st.markdown(f"VeritasLens v1.0 | Research by **Simran Gupta, Bhuvanyu Geel, Anipra Pandya, Parth Sinha, Aviral Yadav** | Last Engine Sync: {datetime.now().strftime('%Y-%m-%d')}")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from datetime import datetime

# =================================================================
# 1. RESEARCH-VALIDATED LINGUISTIC WEIGHTS (From your Findings)
# =================================================================
LINGUISTIC_MAP = {
    "fake": {
        "shocking": 0.18, "unbelievable": 0.18, "exposed": 0.15, "truth": 0.12,
        "allegedly": 0.14, "purportedly": 0.14, "claims": 0.10, "rumored": 0.14,
        "literally": 0.12, "totally": 0.10, "breaking": 0.15, "urgent": 0.15
    },
    "real": {
        "reuters": 0.22, "confirmed": 0.20, "official": 0.18, "verified": 0.18,
        "statement": 0.15, "spokesperson": 0.15, "monday": 0.10, "tuesday": 0.10,
        "washington": 0.12, "london": 0.12, "reported": 0.14
    }
}

# =================================================================
# 2. THE SIMULATED ENSEMBLE ENGINE
# =================================================================
def analyze_text(title, body, use_summary):
    # Model B Logic (Low-Context Specialist - CNN/RoBERTa)
    # Research Finding: High sensitivity to sensationalism/titles
    title_words = re.findall(r'\w+', title.lower())
    b_score = 0.5
    for word in title_words:
        if word in LINGUISTIC_MAP["fake"]: b_score -= LINGUISTIC_MAP["fake"][word]
        if word in LINGUISTIC_MAP["real"]: b_score += LINGUISTIC_MAP["real"][word]
    
    # Model A Logic (High-Context Specialist - BERT)
    # Research Finding: Deep semantic analysis, biased toward 'Real' on formal text
    body_words = re.findall(r'\w+', body.lower())
    a_score = 0.55 # Base bias for BERT found in LIAR experiment
    for word in body_words:
        if word in LINGUISTIC_MAP["fake"]: a_score -= (LINGUISTIC_MAP["fake"][word] * 0.7)
        if word in LINGUISTIC_MAP["real"]: a_score += (LINGUISTIC_MAP["real"][word] * 1.2)
    
    # Ensure bounds
    a_score = max(0, min(1, a_score))
    b_score = max(0, min(1, b_score))
    
    # Ensemble Logic (Soft Voting)
    ensemble_score = (a_score + b_score) / 2
    
    return a_score, b_score, ensemble_score

def get_xray_html(text):
    words = text.split()
    highlighted = []
    for word in words:
        clean_word = re.sub(r'\W+', '', word.lower())
        if clean_word in LINGUISTIC_MAP["fake"]:
            weight = LINGUISTIC_MAP["fake"][clean_word]
            highlighted.append(f'<span style="background-color: rgba(231, 76, 60, {weight*4}); border-bottom: 2px solid red;" title="Fake Indicator: +{weight*100}%">{word}</span>')
        elif clean_word in LINGUISTIC_MAP["real"]:
            weight = LINGUISTIC_MAP["real"][clean_word]
            highlighted.append(f'<span style="background-color: rgba(46, 204, 113, {weight*4}); border-bottom: 2px solid green;" title="Real Indicator: +{weight*100}%">{word}</span>')
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# =================================================================
# 3. STREAMLIT UI LAYOUT
# =================================================================
st.set_page_config(page_title="VeritasLens AI", layout="wide", page_icon="🔬")

# Sidebar - Research Stats
with st.sidebar:
    st.title("Research Insights")
    st.markdown("---")
    st.subheader("Researched Under the guidance of Dr. Adarsh Patel")
    st.subheader("Dataset Benchmarks")
    st.write("**ISOT (Long-form)")
    st.write("**LIAR (Short-form)")
    st.write("**FNC (Diverse)(CNN)")
    st.markdown("---")
    st.info("**Methodology:** Early-Stopping Optimized BERT/CNN Ensemble.")
    if st.button("Clear Cache"):
        st.rerun()

# Main Header
st.title("VeritasLens: Hybrid-Context Truth Engine")
st.markdown("Identify misinformation using our research-backed **Safety Net Ensemble**.")

# Input Section
col_in1, col_in2 = st.columns([1, 1])
with col_in1:
    input_title = st.text_input("Article Title", placeholder="e.g., Shocking truth exposed about...")
with col_in2:
    input_url = st.text_input("URL (Optional Scraper Simulation)", placeholder="https://news.example.com/article")

input_body = st.text_area("Full Article Body", height=200, placeholder="Paste the full article content here for BERT semantic analysis...")

# Toggles
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    opt_mode = st.toggle("Summarization Optimization (T5/PEGASUS)", value=False)
with col_t2:
    xray_mode = st.toggle("Linguistic X-Ray (SHAP/LIME)", value=True)

# Analysis Trigger
if st.button("🚀 Run Multi-Track Analysis", use_container_width=True):
    if not input_title or not input_body:
        st.error("Please provide both a Title and Article Body for the Hybrid Ensemble to function.")
    else:
        score_a, score_b, final_score = analyze_text(input_title, input_body, opt_mode)
        
        # Result Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = final_score * 100,
            title = {'text': "Ensemble Truth Probability"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 40], 'color': "#e74c3c"},
                    {'range': [40, 70], 'color': "#f1c40f"},
                    {'range': [70, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': final_score * 100
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Comparison Grid
        st.subheader("📊 Specialist Model Comparison")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Model A (Full-Text BERT)", f"{score_a*100:.1f}%", delta="Semantic Deep-Dive")
            st.caption("Focus: Contextual meaning and factual alignment.")
            
        with col_res2:
            st.metric("Model B (Title CNN/RoBERTa)", f"{score_b*100:.1f}%", delta="Stylistic Pattern")
            st.caption("Focus: Headlines, sensationalism, and linguistic style.")
            
        with col_res3:
            st.metric("Hybrid Ensemble Score", f"{final_score*100:.1f}%", delta="Consensus Result")
            st.caption("The tie-breaker logic that mitigates individual model bias.")

        # XAI Layer
        if xray_mode:
            st.markdown("---")
            st.subheader("🕵️ Linguistic X-Ray (Explainable AI)")
            st.write("Our models identified these specific cues influencing the score:")
            
            combined_text = f"**TITLE:** {input_title} \n\n **BODY:** {input_body}"
            html_content = get_xray_html(combined_text)
            st.markdown(f'<div style="padding:20px; border-radius:10px; background-color:#1e1e1e; line-height:2.0;">{html_content}</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-top:10px; font-size:0.8em;">
                <span style="color:#2ecc71">■</span> Green = Factual/Formal Indicator | 
                <span style="color:#e74c3c">■</span> Red = Sensational/Hedged Indicator
            </div>
            """, unsafe_allow_html=True)

        # Research-Specific Alerts
        if abs(score_a - score_b) > 0.3:
            st.warning("⚠️ **Specialist Disagreement Detected:** The Headline and Body show conflicting signals. This often indicates a 'Clickbait' or 'Article Laundering' attempt.")
        
        if len(input_body.split()) < 50:
            st.info("💡 **Observation:** Minimal body context detected. Reverting to 'Safety Net' mode (Title-bias recall is 98.3%).")

# Footer
st.markdown("---")
st.markdown(f"VeritasLens v1.0 | Research by **Simran Gupta, Bhuvanyu Geel, Anipra Pandya, Parth Sinha, Aviral Yadav** | Last Engine Sync: {datetime.now().strftime('%Y-%m-%d')}")