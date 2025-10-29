import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import os
import json # Import json for safer parsing
from huggingface_hub import hf_hub_download

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="ResolveAI - MCGM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Assets (Caching for Performance) --- Yes
# We use @st.cache_data for data that doesn't change
@st.cache_data
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        # Convert to datetime (ignoring errors for this prototype)
        df['Created_Date'] = pd.to_datetime(df['Created_Date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {csv_path}")
        return None

# We use @st.cache_resource for ML models and API clients
# @st.cache_resource
# def load_ml_model(model_path):
#     try:
#         model = joblib.load(model_path)
#         return model
#     except FileNotFoundError:
#         st.error(f"Error: Model file not found at {model_path}")
#         return None

# # Load the assets
# DATA_FILE = 'maharashtra_simulated_complaints.csv'
# MODEL_FILE = 'pothole_severity_model.pkl'
# df = load_data(DATA_FILE)
# ml_model = load_ml_model(MODEL_FILE)

@st.cache_resource
def load_ml_model(model_path):
    
    # --- 3. THIS IS THE NEW DOWNLOAD LOGIC ---
    if not os.path.exists(model_path):
        st.info("Downloading 216MB ML model from Hugging Face...")
        st.warning("This may take a moment on first boot.")
        
        try:
            # 4. CONFIGURE YOUR REPO DETAILS
            # Your HF username and the repo name you created
            REPO_ID = "maseerasayed19/pothole-severity-model" 
            # The name of the file you uploaded
            FILENAME = "pothole_severity_model.pkl"

            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=".",             # Download to the current directory
                local_dir_use_symlinks=False, # Recommended for Streamlit
                # 'local_dir' will make it save as './pothole_severity_model.pkl'
                # which matches your 'model_path' variable
            )
            
            st.success("Model downloaded successfully!")
        
        except Exception as e:
            st.error(f"Error downloading model from Hugging Face: {e}")
            st.error("Please double-check your REPO_ID and FILENAME.")
            return None
    # --- END OF NEW LOGIC ---
            
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model after download: {e}")
        return None

# Load the assets
DATA_FILE = 'maharashtra_simulated_complaints.csv'
MODEL_FILE = 'pothole_severity_model.pkl'

df = load_data(DATA_FILE)
# This will now trigger the download logic if needed
ml_model = load_ml_model(MODEL_FILE)


# --- 3. Gemini AI Helper Functions ---
@st.cache_resource
def configure_gemini():
    """Configures the Gemini AI client using the Streamlit secret."""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # --- THE FIX ---
        # Using 'gemini-1.5-flash' is the most robust and modern choice,
        # and it will resolve the 'gemini-pro' 404 error.
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        st.error("Please make sure you have set your GEMINI_API_KEY in .streamlit/secrets.toml")
        return None

def get_genai_response(model, prompt, generation_config=None):
    """
    Gets a response from the Gemini model.
    Accepts an optional generation_config for forcing JSON output.
    """
    if model is None:
        return "Error: Gen AI Model not configured."
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config # Pass the config here
        )
        return response.text
    except Exception as e:
        # Return the specific error message from the API
        return f"An error occurred during API call: {e}"

# Configure the Gen AI model
gemini_model = configure_gemini()

# Lists for dropdowns (from your research)
mumbai_wards_list = [
    'A Ward (Colaba/Churchgate)', 'B Ward (Dongri/Masjid Bunder)', 
    'C Ward (Bhuleshwar/Pydhonie)', 'D Ward (Grant Road/Malabar Hill)',
    'E Ward (Byculla/Chinchpokli)', 'F/North Ward (Matunga/Sion)', 
    'F/South Ward (Parel/Sewri)', 'G/North Ward (Dadar/Mahim)', 
    'G/South Ward (Worli/Lower Parel)', 'H/East Ward (Bandra East)', 
    'H/West Ward (Bandra West)', 'K/East Ward (Andheri East)', 
    'K/West Ward (Andheri West)', 'L Ward (Kurla)', 
    'M/East Ward (Chembur/Govandi)', 'M/West Ward (Chembur West)', 
    'N Ward (Ghatkopar/Vikhroli)', 'P/North Ward (Malad)', 
    'P/South Ward (Goregaon)', 'R/Central Ward (Borivali)', 
    'R/North Ward (Dahisar)', 'R/South Ward (Kandivali)', 
    'S Ward (Bhandup/Powai)', 'T Ward (Mulund)'
]
local_issue_types = ['Pothole/Crater', 'Water Pipeline Leakage', 'Drainage Overflow', 'Road Surface Erosion']

# --- 4. Mock Login Page ---
def show_login_page():
    # We must set page_config ONLY ONCE, at the top of the script.
    # So we remove it from here and just use layout commands.
    st.title("ResolveAI - AI-Powered Governance Platform")
    st.header("MCGM (Mumbai) Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # Simple mock login for the hackathon
            if username == "person" and password == "person123":
                st.session_state["logged_in"] = True
                # 'st.experimental_rerun()' is deprecated. Use 'st.rerun()'.
                st.rerun()
            else:
                st.error("Invalid username or password")

# --- 5. Main Application ---
def show_main_app():
    # Sidebar Navigation
    st.sidebar.title("ResolveAI 🤖")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", [
        "M-Gov Dashboard", 
        "Analyze New Complaint", 
        "Proactive Resolution"
    ])
    
    st.sidebar.info("A prototype for the State Government of Maharashtra.")

    # --- Page 1: M-Gov Dashboard ---
    if page == "M-Gov Dashboard":
        st.title("M-Gov Predictive Dashboard (MCGM)")
        
        if df is not None:
            # KPIs
            total_complaints = len(df)
            high_severity_alerts = int(df['Severity'].sum())
            wards_affected = df['Ward'].nunique()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Open Complaints", f"{total_complaints:,}")
            col2.metric("High-Severity Alerts", f"{high_severity_alerts:,}")
            col3.metric("Wards Affected", wards_affected)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Complaints by Ward")
                ward_counts = df['Ward'].value_counts().reset_index()
                fig_bar = px.bar(ward_counts.head(10), x='Ward', y='count', title="Top 10 Wards by Complaint Volume")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.subheader("Issue Type Breakdown")
                issue_counts = df['Issue_Type'].value_counts().reset_index()
                fig_pie = px.pie(issue_counts, names='Issue_Type', values='count', title="Complaint Types")
                st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()
            
            # Map
            st.subheader("Live Issue Hotspot Map")
            st.warning("Displaying a random sample of 1,000 high-severity issues for performance.")
            
            # Sample the data to avoid crashing Folium/browser
            # Add a check in case there are less than 1000 alerts
            sample_size = min(1000, high_severity_alerts)
            if sample_size > 0:
                df_sample = df[df['Severity'] == 1].sample(sample_size)
                
                # Create a Folium map centered on Mumbai
                map_center = [19.0760, 72.8777] # Approx. center of Mumbai
                m = folium.Map(location=map_center, zoom_start=11)
                
                # Add points to the map
                for _, row in df_sample.iterrows():
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=3,
                        color='red',
                        fill=True,
                        fill_color='red',
                        popup=f"<strong>{row['Issue_Type']}</strong><br>{row['Ward']}"
                    ).add_to(m)
                
                # Render the map in Streamlit
                st_folium(m, width=700, height=500, returned_objects=[])
            else:
                st.info("No high-severity alerts to display on the map.")
        
        else:
            st.error("Data could not be loaded. Dashboard cannot be displayed.")

    # --- Page 2: Analyze New Complaint (Prioritize) ---
    elif page == "Analyze New Complaint":
        st.title("Analyze New Citizen Complaint")
        st.info("This page uses Gen AI (Gemini) to parse unstructured citizen complaints and convert them into actionable intelligence.")
        
        complaint_text = st.text_area("Paste Raw Complaint Text Here:", height=150, 
                                      placeholder="e.g., 'There's a massive, deep pothole on the road in K/West Ward near Andheri station. My car's tire is gone. It's an emergency.'")
        
        if st.button("Analyze Complaint"):
            if gemini_model is None:
                st.error("Gen AI Model is not ready. Check your API key.")
            elif not complaint_text:
                st.warning("Please enter some complaint text to analyze.")
            else:
                # This is your "Prompt 1"
                prompt = f"""
                You are ResolveAI, an expert AI analyst for the MCGM (Mumbai) Public Works Department. 
                Your job is to process raw citizen complaints and output structured JSON.

                When you receive a complaint, you MUST return *only* a JSON object with the following keys:
                - "issue_type": (Must be one of: 'Pothole/Crater', 'Water Pipeline Leakage', 'Drainage Overflow', 'Road Surface Erosion')
                - "urgency": (Must be one of: "Low", "Medium", "High")
                - "summary": (A 1-2 sentence summary of the citizen's report)
                - "suggested_action": (A brief recommended next step for the city manager)

                Here is the complaint:
                "{complaint_text}"
                """
                
                # --- NEW: Force the model to output JSON ---
                json_generation_config = genai.GenerationConfig(
                    response_mime_type="application/json"
                )
                
                with st.spinner("🤖 AI is analyzing the complaint..."):
                    response_text = get_genai_response(
                        gemini_model, 
                        prompt, 
                        generation_config=json_generation_config
                    )
                    
                    # Robust JSON Parsing
                    if response_text.strip().startswith("An error"):
                        st.error(response_text)
                    else:
                        st.subheader("AI Analysis Results")
                        try:
                            # Clean the response (though JSON mode should be clean)
                            json_response = response_text.strip().replace("```json", "").replace("```", "")
                            data = json.loads(json_response) 
                            
                            # --- NEW: Display results in a clean UI ---
                            urgency = data.get("urgency", "Unknown")
                            delta_color = "normal"
                            delta_text = ""
                            if urgency == "High":
                                delta_color = "inverse"
                                delta_text = "🔴 High"
                            elif urgency == "Medium":
                                delta_color = "inverse"
                                delta_text = "🟠 Medium"
                            elif urgency == "Low":
                                delta_text = "🟢 Low"
                                
                            st.metric("Urgency", urgency, delta_text, delta_color=delta_color)
                            
                            st.info(f"**Identified Issue Type:** {data.get('issue_type', 'N/A')}")
                            
                            st.warning(f"**Suggested Action:** {data.get('suggested_action', 'N/A')}")
                            
                            st.subheader("Summary of Complaint")
                            st.write(data.get('summary', 'N/A'))

                            with st.expander("Show Raw AI Response (JSON)"):
                                st.json(data)

                        except Exception as e:
                            st.error(f"Failed to parse AI response as JSON: {e}")
                            st.subheader("Raw AI Response (that failed parsing):")
                            st.code(response_text)
    
    # --- Page 3: Proactive Resolution (Resolve) ---
    elif page == "Proactive Resolution":
        st.title("Proactive Work Order Resolution")
        st.info("This page uses Gen AI (Gemini) to draft a professional work order for a predicted issue *before* it gets reported by a citizen.")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_ward = st.selectbox("Select Ward:", options=mumbai_wards_list)
        with col2:
            selected_issue = st.selectbox("Select Predicted Issue:", options=local_issue_types)

        if st.button("Draft Proactive Work Order"):
            if gemini_model is None:
                st.error("Gen AI Model is not ready. Check your API key.")
            else:
                # This is your "Prompt 2"
                prompt = f"""
                You are a Work Order Generator for the MCGM. Given structured data about an infrastructure issue, 
                draft a professional work order for a field crew.

                Here is the data:
                - "Ward": "{selected_ward}"
                - "Issue": "{selected_issue}"
                - "AI_Severity": "High"
                - "Source": "ResolveAI Predictive Model (Hotspot)"
                """
                
                with st.spinner("🤖 AI is drafting the work order..."):
                    response_text = get_genai_response(gemini_model, prompt)
                    
                    st.subheader("Drafted Work Order")
                    # Check for errors before displaying
                    if response_text.strip().startswith("An error"):
                        st.error(response_text)
                    else:
                        st.markdown(response_text)


# --- 6. App Controller (Login Check) ---
# This part handles the session state to show/hide the app
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    # If logged in, run the main app logic
    show_main_app()
else:
    # If not logged in, show the login page
    show_login_page()





#There's a massive deep pothole on the road in  H/West ward near Bandra Station. My Car's tire is gone. It's an emergency.
# In P/North Ward, near the Malad fire station, a water pipe seems to have burst and is flooding the sidewalk. It's been running for two hours now and is just a big mess.
# The road surface in S Ward (Bhandup) is getting very cracked and uneven along the main stretch. It's not a pothole yet, but the asphalt is breaking up badly. It needs maintenance soon.
# There's a terrible smell in G/South Ward near the Worli market. The gutter seems blocked, and dirty water is overflowing onto the street when it rains, making the area unusable.
# We have a large pothole forming on a side street in L Ward (Kurla). It's manageable for now, but if it rains again tonight, it will become a serious hazard and needs a patch job by tomorrow.
#An error occurred during API call: 404 models/gemini-pro is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods.