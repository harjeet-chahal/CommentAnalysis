
import streamlit as st
import requests
import pandas as pd
import time
import random
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")


# Page Config
st.set_page_config(
    page_title="Sentinel: AI Content Moderation",
    page_icon="🛡️",
    layout="centered"
)

# Title & Description
st.title("🛡️ Sentinel")
st.header("AI Content Moderation System")
st.markdown("Enter a comment below to analyze its toxicity levels using the **Tier 3 RoBERTa** model.")


# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Moderator Tool", "Analytics"])

if page == "Moderator Tool":
    # Sidebar Config
    st.sidebar.header("Configuration")
    sensitivity = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05, help="Override the default flags. Lower values trigger flags more easily.")

    use_mock_backend = st.sidebar.checkbox("Use Mock Backend (Demo Mode)", value=False, help="Check this if the API is offline.")

    # Mock Logic
    def mock_predict(text):
        time.sleep(1.0) # Simulate latency
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        scores = {l: random.random() for l in labels}
        
        # Simple logic for demo: if text contains 'hate', make it toxic
        if "hate" in text.lower() or "bad" in text.lower():
            scores['toxic'] = 0.95
        else:
            scores['toxic'] = 0.05
            
        return scores

    # Main Interface
    st.header("Moderator Tool")
    comment = st.text_area("User Comment", height=150, placeholder="Type your comment here...")

    if st.button("Analyze", type="primary"):
        if not comment.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing content..."):
                scores = {}
                error_msg = None
                
                if use_mock_backend:
                    scores = mock_predict(comment)
                else:
                    try:
                        response = requests.post(BACKEND_URL, json={"text": comment}, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            scores = data['scores']
                        else:
                            error_msg = f"API Error: {response.text}"
                    except requests.exceptions.ConnectionError:
                        error_msg = "Could not connect to backend. Is it running? (Try 'Mock Backend' in sidebar)"
                    except Exception as e:
                        error_msg = f"An error occurred: {e}"
                
                # Display Results
                if error_msg:
                    st.error(error_msg)
                else:
                    # Determine Identity based on Sensitivity Slider
                    flags = [label for label, score in scores.items() if score >= sensitivity]
                    
                    # Visuals
                    if flags:
                        st.error("🚩 **Toxic Content Detected**")
                        st.markdown(f"**Flags Triggered**: {', '.join([f'`{f}`' for f in flags])}")
                    else:
                        st.success("🛡️ **Content is Safe**")
                        st.markdown("No flags triggered based on current sensitivity.")
                    
                    # Bar Chart
                    st.subheader("Probability Breakdown")
                    chart_data = pd.DataFrame(index=scores.keys(), data=scores.values(), columns=["Probability"])
                    st.bar_chart(chart_data)
                    
                    # JSON view
                    with st.expander("View Raw JSON"):
                        st.json(scores)
                        
                    # Explainability
                    if st.button("Explain Why (LIME)"):
                        with st.spinner("Generating explanation (this may take a few seconds)..."):
                            if use_mock_backend:
                                st.warning("LIME visualization requires the live backend.")
                            else:
                                try:
                                    exp_response = requests.post(
                                        f"{BACKEND_URL.replace('/predict', '/explain')}", 
                                        json={"text": comment}, 
                                        timeout=30
                                    )
                                    if exp_response.status_code == 200:
                                        html_content = exp_response.json()['html']
                                        st.components.v1.html(html_content, height=800, scrolling=True)
                                    else:
                                        st.error(f"Explanation Error: {exp_response.text}")
                                except Exception as e:
                                    st.error(f"Failed to get explanation: {e}")

                    # Feedback Loop
                    with st.expander("Report Error / Provide Feedback"):
                        st.write("Disagree with the prediction? Let us know.")
                        
                        # Pre-select current flags
                        current_flags = [label for label, score in scores.items() if score >= sensitivity]
                        
                        all_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                        suggested = st.multiselect("Correct Labels", options=all_labels, default=current_flags)
                        
                        user_note = st.text_area("Optional Note", placeholder="Why is the model wrong?")
                        
                        if st.button("Submit Feedback"):
                            if use_mock_backend:
                                st.success("Feedback simulated (Mock Mode).")
                            else:
                                try:
                                    fb_response = requests.post(
                                        f"{BACKEND_URL.replace('/predict', '/feedback')}", 
                                        json={
                                            "text": comment,
                                            "suggested_labels": suggested,
                                            "user_comment": user_note
                                        }, 
                                        timeout=5
                                    )
                                    if fb_response.status_code == 200:
                                        st.success("Thank you! This will be used to improve the model.")
                                    else:
                                        st.error(f"Error submitting feedback: {fb_response.text}")
                                except Exception as e:
                                    st.error(f"Submission failed: {e}")

elif page == "Analytics":
    st.header("Admin Analytics")
    feedback_file = "data/feedback/corrections.csv"
    
    if not os.path.exists(feedback_file):
        st.info("No feedback data available yet.")
    else:
        try:
            df = pd.read_csv(feedback_file)
            st.metric("Total User Corrections", len(df))
            
            # Label Analysis
            st.subheader("Missed Labels Frequency")
            if not df.empty:
                # Count individual labels from pipe-separated string
                all_labels = []
                for labels_str in df['suggested_labels'].dropna():
                    if labels_str:
                        all_labels.extend(labels_str.split('|'))
                
                if all_labels:
                    label_counts = pd.Series(all_labels).value_counts()
                    st.bar_chart(label_counts)
                    st.caption("Which labels users say the model missed (or incorrectly flagged).")
                else:
                    st.write("No labels provided in feedback.")
                
                # Recent Feedback
                st.subheader("Recent Submissions")
                st.dataframe(df.tail(10)[::-1]) # Reverse to show newest first
            else:
                st.write("Feedback file is empty.")
                
        except Exception as e:
            st.error(f"Error reading analytics data: {e}")


# Footer
st.markdown("---")
st.markdown("Powered by RoBERTa · Built with FastAPI & Streamlit")
