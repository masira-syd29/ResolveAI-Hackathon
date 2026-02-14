# ResolveAI
**Hybrid AI-Driven Predictive Governance for Urban Infrastructure**

ResolveAI is a sophisticated decision-support system designed for the **Municipal Corporation of Greater Mumbai (MCGM)**. It bridges the gap between raw citizen grievances and proactive government action by combining traditional Machine Learning with Large Language Models (LLMs).

🔗 **Live Demo**: [https://resolveai-hackathon-29ms.streamlit.app/](https://resolveai-hackathon-29ms.streamlit.app/)

---

## 📸 System Overview


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