import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- 1. Define Mumbai Context (Using YOUR awesome research!) ---
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
print("--- Step 1: Context Defined ---")
print(f"Loaded {len(mumbai_wards_list)} Mumbai Wards.")

# --- 2. Load & Filter NYC Data (Memory Efficiently) ---
NYC_DATA_FILE = '311_Service_Requests_from_2010_to_Present.csv'
print(f"\n--- Step 2: Loading Raw Data in CHUNKS ({NYC_DATA_FILE}) ---")

# Define the columns we need to load
columns_to_load = ['Complaint Type', 'Descriptor', 'Latitude', 'Longitude', 'Created Date']

# Define the chunksize
chunk_size = 1_000_000 
# 1 million rows at a time
filtered_chunks = [] 
# We will store our filtered data here
chunk_num = 1

try:
    with pd.read_csv(NYC_DATA_FILE,
                     usecols=columns_to_load,
                     low_memory=False,
                     chunksize=chunk_size) as reader:

        for chunk in reader:
            print(f"Processing chunk {chunk_num}...")
            
            # --- THIS IS YOUR NEW, PRECISE FILTER ---
            # Filter 1: Potholes (based on your new info)
            filter_potholes = (chunk['Complaint Type'] == 'Street Condition') & \
                              (chunk['Descriptor'] == 'Pothole')
            
            # Filter 2: Water Leaks (another good category to include)
            filter_water = chunk['Complaint Type'] == 'Water System'
            
            # Combine the filters
            filtered_chunk = chunk[filter_potholes | filter_water].copy()
            
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
                print(f"  > Found {len(filtered_chunk)} relevant rows in this chunk.")
                
            chunk_num += 1

    print("\nFinished processing all chunks.")
    
    # Check if we found any data at all
    if not filtered_chunks:
        print("!!! ERROR: No relevant data found with the current filters.")
        print("Please double-check the 'Complaint Type' and 'Descriptor' names.")
        exit()

    # Combine all the small filtered chunks into one DataFrame
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Total relevant rows found: {len(df_filtered)}")

except FileNotFoundError:
    print(f"!!! ERROR: File not found: '{NYC_DATA_FILE}'")
    exit()
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# --- 3. Clean Data ---
print("\n--- Step 3: Cleaning Data ---")
# Now we can proceed with the small, filtered DataFrame
df_cleaned = df_filtered.dropna(subset=['Descriptor', 'Latitude', 'Longitude', 'Created Date']).copy()

if len(df_cleaned) == 0:
    print("!!! ERROR: All relevant rows were dropped due to missing data (NaN).")
    exit()

print(f"Total clean rows ready for simulation: {len(df_cleaned)}")

# --- 4. Localize the Data (The "Mumbai Hack") ---
print("\n--- Step 4: Localizing Data ---")
df_simulated = pd.DataFrame()
df_simulated['Description'] = df_cleaned['Descriptor']
df_simulated['Latitude'] = df_cleaned['Latitude']
df_simulated['Longitude'] = df_cleaned['Longitude']
df_simulated['Created_Date'] = pd.to_datetime(df_cleaned['Created Date'])
df_simulated['Ward'] = np.random.choice(mumbai_wards_list, size=len(df_simulated))
df_simulated['Issue_Type'] = np.random.choice(local_issue_types, size=len(df_simulated))
print("Applied Mumbai wards and issue types.")

# --- 5. Feature Engineering & Target Creation ---
print("\n--- Step 5: Engineering Features ---")
high_severity_keywords = 'large|deep|dangerous|massive|hazard|unsafe|emergency|burst|overflow|leak'
df_simulated['Severity'] = df_simulated['Description'].str.contains(high_severity_keywords, case=False, na=False).astype(int)
df_simulated['mock_road_age_years'] = np.random.randint(2, 35, size=len(df_simulated))
df_simulated['mock_recent_precipitation_in'] = np.random.uniform(0.0, 5.0, size=len(df_simulated))
df_simulated['Month'] = df_simulated['Created_Date'].dt.month
print("Created 'Severity' target and 'mock' features.")

# --- 6. Save the Clean, Localized Data ---
SIMULATED_CSV_FILE = 'maharashtra_simulated_complaints.csv'
df_simulated.to_csv(SIMULATED_CSV_FILE, index=False)
print(f"\n--- Step 6: Clean, Localized Data Saved ---")
print(f"Successfully saved to '{SIMULATED_CSV_FILE}'")
print(df_simulated.head())

# --- 7. Train the Predictive Model (Traditional ML) ---
print("\n--- Step 7: Training Predictive Model ---")
features = ['Latitude', 'Longitude', 'Month', 'mock_road_age_years', 'mock_recent_precipitation_in']
target = 'Severity'
X = df_simulated[features]
y = df_simulated[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- REVERTED TO ORIGINAL HIGH-ACCURACY SETTINGS ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# ---------------------------------------------------

print("Fitting RandomForest model...")
model.fit(X_train, y_train)
print("Model fitting complete.")
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- 8. Save the Trained Model ---
MODEL_FILE = 'pothole_severity_model.pkl'
joblib.dump(model, MODEL_FILE) # Save the large model
print(f"\n--- Step 8: Model Saved ---")
print(f"Predictive model saved to '{MODEL_FILE}'")
print("\n--- PHASE 1 (STEP 3) COMPLETE! ---")

# https://www.kaggle.com/datasets/josefsieber/311-service-requests-from-2010-to-present?resource=download