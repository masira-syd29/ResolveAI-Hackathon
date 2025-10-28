ResolveAI: AI-Powered Governance for MCGM (Mumbai)

Predict. Prioritize. Resolve. Before it escalates.

This is a prototype solution for the Gen AI Exchange Hackathon (Wild Card Challenge), submitted for the "AI-Powered Governance" problem statement contributed by the State Government of Maharashtra.

🎥 [Link to 3-Minute Demo Video]

(You will add this link here after you record it)

🚀 [Link to Live Working Prototype]

(Add your Streamlit Cloud URL here after you deploy)

1. The Problem

Government bodies like the MCGM (Mumbai) face a constant flood of unstructured citizen complaints (311 calls, social media posts) for public works issues like potholes and water leaks. By the time an issue is reported, it has often already caused damage or disruption. The challenge is to move from a reactive to a proactive governance model.

2. Our Solution: ResolveAI

ResolveAI is a secure, AI-driven platform for MCGM decision-makers that transforms raw government data into actionable intelligence. It uses a Dual-AI System to manage and predict public works failures.

The Dual-AI System

Predictive AI (ML): A RandomForestClassifier trained on 513,000+ real-world complaint data points (localized to all 24 Mumbai Wards) to predict the severity of emerging issues. Our model achieved ~80% accuracy in identifying high-severity risks.

Generative AI (Gen AI): A gemini-1.5-flash model that acts as a 24/7 AI analyst to:

Prioritize: Instantly read, classify, and summarize new, unstructured citizen complaints into structured, actionable JSON.

Resolve: Proactively draft professional work orders for field crews to fix AI-predicted hotspots before they are ever reported by a citizen.

3. Key Features

M-Gov Predictive Dashboard: A live map of high-severity hotspots across all 24 MCGM Wards, with interactive charts for "Complaints by Ward" and "Issue Type Breakdown."

AI-Powered Triage ("Analyze New Complaint"): A tool for managers to paste in any raw complaint and get an instant, user-friendly analysis of its urgency, type, and a suggested action.

Proactive Resolution ("Proactive Resolution"): A "Work Order" generator that drafts professional, ready-to-send resolutions for AI-predicted issues, saving time and resources.

Secure Access: A role-based login (prototype-level) to ensure data is seen only by authorized personnel.

4. Tech Stack

Core Application: Streamlit (deployed on Streamlit Cloud)

Predictive AI: Python, Pandas, Scikit-learn, Joblib

Generative AI: Google Gemini 2.5 Flash (via google-generativeai)

Data Visualization: Plotly, Folium (for interactive maps)

5. How to Run This Project Locally

Clone the repository:

git clone [https://github.com/masira-syd29/ResolveAI-Hackathon]


Create and activate a virtual environment:

# On Windows
python -m venv venv
.\venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Set up your API key:

Create a file at this exact path: .streamlit/secrets.toml

Add your Google AI (Gemini) API key to it:

GEMINI_API_KEY = "YOUR_API_KEY_GOES_HERE"


Run the app:

streamlit run app.py