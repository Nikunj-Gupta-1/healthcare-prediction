# healthcare-prediction

An interactive **Streamlit web dashboard** for training and evaluating machine learning models on synthetic healthcare patient datasets. Built during **Practice School I (PS-1)** as a parallel project alongside the DDoS detection work.

---

## How This Fits Into the Larger Work

This is a **standalone project** that runs independently from the DDoS detection series:

```
PS1-coursework-experiments         (early classifiers and Scapy experiments)
PS1-monolithic-ddos-detection      (single-machine DDoS sniffer)
PS1-distributed-ddos-detection     (distributed DDoS pipeline)

healthcare-prediction              ← you are here (separate, parallel project)
```

The DDoS detection series and this project share the same PS-1 timeframe and some common ML tooling (scikit-learn, pandas), but address entirely different problem domains.

---

## What the Project Does

- **Generates synthetic patient data** based on configurable distributions (age, billing, medical condition, etc.)
- **Trains a Random Forest binary classifier** to predict high-risk patients
- **Visualises model performance** — confusion matrix, ROC curve, feature importances — directly in the browser
- **Supports multiple distribution types** — uniform, normal, exponential — for flexible data generation

---

## What's in This Repo

| File | Description |
|---|---|
| `modeltrainer.py` | **Main entrypoint.** Full Streamlit dashboard with configurable columns, risk weights, and distribution types |
| `app.py` | Earlier version of the dashboard with a simpler configuration interface |
| `selfdata.py` | Core module: data generation, preprocessing, and Random Forest training logic |

### Documents (`documents/`)
| Subfolder | Contents |
|---|---|
| `documents/reports/` | PS-1 midsem report (covers this project), final PS-1 report |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

**Launch the dashboard:**
```bash
streamlit run modeltrainer.py
```

Open `http://localhost:8501` in your browser.

---

## Data

This project generates **fully synthetic data at runtime** — no external datasets are required.
No `.csv` or model files are included in this repo.

---

## See Also
- [`PS1-coursework-experiments`](https://github.com/nikunj-gupta-1/PS1-coursework-experiments) — early PS-1 ML assignments
- [`PS1-monolithic-ddos-detection`](https://github.com/nikunj-gupta-1/PS1-monolithic-ddos-detection) — DDoS detection (monolithic)
- [`PS1-distributed-ddos-detection`](https://github.com/nikunj-gupta-1/PS1-distributed-ddos-detection) — DDoS detection (distributed pipeline)
