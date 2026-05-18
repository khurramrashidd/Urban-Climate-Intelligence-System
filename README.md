# 🌆 Urban Climate Intelligence System

A sophisticated **AI/ML-powered Urban Heat Island & Air Quality Predictor** built in Google Colab that fetches real-time weather and pollution data, runs a **Stacked Ensemble ML model**, explains predictions using **SHAP**, and generates intelligent **LLM-powered health advisories** via Llama 3.

This project demonstrates end-to-end ML engineering — from real API data ingestion and physics-based feature engineering to model explainability and an interactive Gradio dashboard.

---

## 🚀 Features

* Real-time weather & air pollution data via OpenWeatherMap API
* 30+ engineered features (Heat Index, Dew Point, Wet Bulb Temp, UHI Proxy, Pollution Composite, etc.)
* Stacked Ensemble ML model (XGBoost + Random Forest + Gradient Boosting → Logistic Regression)
* Secondary XGBoost Heat Stress Classifier (5 NOAA-based levels)
* SHAP TreeExplainer for per-instance feature attribution and explainability
* LLM-generated health advisories using **Llama 3.3 (Groq API)** — free tier
* Multi-city comparison with interactive Plotly radar and bar charts
* 5-day forecast trend analysis with predicted AQI timeline
* Live interactive **Gradio dashboard** with public shareable URL

---

## 🛠 Built With

* **Python 3.12** (Google Colab)
* **XGBoost** — gradient boosted trees
* **Scikit-learn** — Random Forest, Gradient Boosting, Stacking, StandardScaler
* **SHAP** — model explainability
* **Groq SDK** — Llama 3.3-70b-versatile (LLM inference)
* **OpenWeatherMap API** — weather, air pollution, forecast
* **Gradio** — interactive web dashboard
* **Plotly & Matplotlib** — visualizations
* **Pandas & NumPy** — data engineering

---

## 📂 Project Structure

```text
urban-climate-intelligence/
│
├── README.md
│
└── colab_notebook/
    ├── Cell 1  — API Keys (OWM + Groq)
    ├── Cell 2  — Install Dependencies
    ├── Cell 3  — Imports
    ├── Cell 4  — Data Fetching Functions (OWM API)
    ├── Cell 5  — Feature Engineering (30+ features)
    ├── Cell 6  — Multi-City Real-Time Data Collection
    ├── Cell 7  — Synthetic Data Augmentation (~4000 samples)
    ├── Cell 8  — Stacked Ensemble Training + Evaluation
    ├── Cell 9  — Heat Stress XGBoost Classifier
    ├── Cell 10 — SHAP Explainability + Summary Plot
    ├── Cell 11 — LLM Advisory Generator (Groq + Llama 3)
    ├── Cell 12 — Full City Prediction Pipeline
    ├── Cell 13 — Multi-City Comparison Dashboard (Plotly)
    └── Cell 14 — Gradio Interactive Web App
```

---

## 🔑 API Keys Required (Both Free)

| API | Purpose | Free Tier |
|-----|---------|-----------|
| [OpenWeatherMap](https://openweathermap.org/api) | Weather + Air Pollution + Forecast | 1,000 calls/day |
| [Groq](https://console.groq.com) | Llama 3.3 LLM inference | Free tier available |

---

## ⚙️ Installation

### 1. Open Google Colab

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Get your API Keys

**OpenWeatherMap:**
```
https://home.openweathermap.org/users/sign_up
→ After signup → API Keys tab → copy default key
```

**Groq:**
```
https://console.groq.com
→ After signup → API Keys → Create new key
```

### 3. Paste API Keys in Cell 1

```python
OWM_API_KEY  = "your_openweathermap_key_here"
GROQ_API_KEY = "your_groq_key_here"
```

---

## ▶️ Usage

Paste all 14 cells into Colab in order and run them sequentially, or use **Runtime → Run All**.

### Predict any city:
```python
predict_city("Mumbai")
predict_city("London")
predict_city("Tokyo")
```

### Launch the interactive dashboard:
Cell 14 automatically launches Gradio and prints a **public shareable URL** — open it in any browser, type any city name, and click Analyze.

---

## 🧠 ML Architecture

```text
Input (30+ features)
        │
        ├──► XGBoost Classifier ──────────┐
        ├──► Random Forest Classifier ────┼──► Logistic Regression (Meta) ──► AQI Prediction (1–5)
        └──► Gradient Boosting Classifier ┘

Input (13 thermal features)
        │
        └──► XGBoost Classifier ──────────────────────────────────────────► Heat Stress Level (0–4)

Predictions ──► SHAP TreeExplainer ──► Top feature drivers per instance
             ──► Groq Llama 3.3    ──► Natural language health advisory
```

---

## 📊 Features Engineered

| Category | Features |
|----------|----------|
| **Raw Weather** | Temperature, Feels Like, Humidity, Pressure, Wind Speed, Cloud Cover, Visibility |
| **Raw Pollution** | PM2.5, PM10, NO2, O3, CO, SO2 |
| **Derived Thermal** | Heat Index (Rothfusz), Dew Point (Magnus), Wet Bulb Temp (Stull), Temp Range, Feels Delta |
| **Atmospheric** | Pressure Anomaly, Wind U/V Components |
| **Pollution Composite** | Weighted composite score, PM ratio, NOx level, Oxidant level |
| **Urban Heat** | UHI Proxy, Thermal Comfort Index |
| **Condition Flags** | Is Precipitation, Is Fog, Is Clear, Visibility Stress |

---

## 🔐 Best Practices Followed

* Physics-based feature derivation (not just raw API values)
* Stratified train/test split to handle class imbalance
* Stacking with cross-validated base learners to prevent leakage
* SHAP for transparent, auditable predictions
* Graceful error handling in all API calls and SHAP extraction
* Safe SHAP shape handling across different library versions

---

## 🔮 Future Improvements

* Add satellite-derived NDVI (vegetation index) as UHI feature
* Time-series LSTM for 7-day AQI forecasting
* Integrate NASA FIRMS API for wildfire smoke contribution
* Add user health profile (age, respiratory conditions) to personalize advisories
* Deploy as a FastAPI backend + React frontend web app
* Add map-based city picker using Folium or Leaflet

---

## 📜 License

This project is open source and available under the **MIT License**.

---

## 👨‍💻 Author

**Khurram Rashid**
B.Tech Computer Science Engineering
Amity University Mumbai
