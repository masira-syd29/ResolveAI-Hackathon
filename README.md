# ResolveAI
**Hybrid AI-Driven Predictive Governance for Urban Infrastructure**

ResolveAI is a sophisticated decision-support system designed for the **Municipal Corporation of Greater Mumbai (MCGM)**. It bridges the gap between raw citizen grievances and proactive government action by combining traditional Machine Learning with Large Language Models (LLMs).

🔗 **Live Demo**: [https://resolveai-hackathon-29ms.streamlit.app/](https://resolveai-hackathon-29ms.streamlit.app/)

<<<<<<< HEAD
---

## 📸 System Overview
=======
🚀(https://resolveai-hackathon-29ms.streamlit.app/)

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
>>>>>>> 3b9823b612b565f16b455b9029862e60293a2ac0


---

## 🚩 The Problem
Government bodies like the MCGM face a constant flood of unstructured citizen complaints (311 calls, social media, emails). By the time an issue is reported, it has often already caused significant damage or public risk. ResolveAI shifts the paradigm from **Reactive** to **Proactive** governance.

---

## 🚀 Key Engineering Features

### 1. Dual-AI Predictive Engine
* **Predictive AI (ML):** Utilizes a `RandomForestClassifier` trained on **513,000+ real-world data points** (localized to all 24 Mumbai Wards). 
* **Performance:** Achieved **~80% accuracy** in identifying high-severity risks.
* **Logic:** Calculates a Severity Score ($S$) based on geospatial and environmental features:
    $$S = f(\text{Latitude, Longitude, RoadAge, Precipitation})$$

### 2. Intelligent Complaint Parsing (NLP)
* **Generative AI:** Leverages **Gemini 1.5 Flash** to transform unstructured, multi-lingual citizen text into structured JSON.
* **Extraction:** Achieves 95%+ accuracy in identifying issue types, urgency levels, and situational summaries.

### 3. Geospatial Intelligence
* **Hotspot Mapping:** Real-time visualization of high-priority issues using `Folium` and Coordinate Geometry.
* **Dynamic Analytics:** Live M-Gov Dashboard displaying complaint volume by Ward and issue breakdown.

### 4. Automated Workflow Orchestration
* **Proactive Resolution:** Automatically drafts professional, department-ready Work Orders (WO) for field crews based on AI-predicted hotspots before they are even reported by citizens.

---

## 🛠️ Tech Stack

* **Core Logic:** Python 3.10+
* **Generative AI:** Google Gemini API (NLP & JSON Extraction)
* **Machine Learning:** Scikit-Learn (RandomForest), Joblib, Hugging Face Hub (Model Versioning)
* **Data Orchestration:** Pandas, NumPy
* **Visualization:** Plotly Express (Analytics), Folium (GIS Mapping)
* **Deployment:** Streamlit Cloud

---

## ⚙️ Setup & Installation

<<<<<<< HEAD
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/masira-syd29/ResolveAI-Hackathon](https://github.com/masira-syd29/ResolveAI-Hackathon)
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Secrets:**
    Create `.streamlit/secrets.toml` and add your Google AI Key:
    ```toml
    GEMINI_API_KEY = "YOUR_API_KEY"
    ```

4.  **Launch Application:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Data Source
The model is trained on a localized version of the **311 Service Requests dataset** (4.67GB).
[Explore Dataset on Kaggle](https://www.kaggle.com/datasets/josefsieber/311-service-requests-from-2010-to-present)
=======
6. You will find the Kaggle Dataset here (4.67GB)
[https://www.kaggle.com/datasets/josefsieber/311-service-requests-from-2010-to-present?resource=download]
>>>>>>> 3b9823b612b565f16b455b9029862e60293a2ac0
