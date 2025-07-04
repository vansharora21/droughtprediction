# 🌾 Drought Prediction using Environmental and Rainfall Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project predicts drought conditions using rainfall, soil moisture, and other environmental variables. It uses machine learning models including **XGBoost** and **LSTM** to provide reliable early-warning signals for droughts.

---

## 📁 Folder Structure

```
droughtprediction/
├── data/           # Input datasets (CSV format)
├── models/         # Trained and serialized ML models
├── notebooks/      # Jupyter notebooks for EDA and modeling
├── results/        # Graphs, metrics, and evaluation outputs
├── scripts/        # Python scripts for preprocessing and training
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vansharora21/droughtprediction.git
cd droughtprediction
```

### 2. Install Dependencies
We recommend using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
Launch the interactive notebook for exploration and training:

```bash
jupyter notebook notebooks/DroughtPrediction.ipynb
```

### 4. Or Run the Training Script
```bash
python scripts/train_model.py
```

> For a more dynamic version, you can extend `train_model.py` to accept command-line arguments using `argparse`.

---

## 🤖 Models Used

| Model     | Description                                     |
|-----------|-------------------------------------------------|
| XGBoost   | Gradient boosting model for tabular data        |
| LSTM      | Recurrent neural network for time-series data   |

---

## 📊 Output and Results

- All evaluation metrics, graphs, and plots are saved in the `results/` directory.
- Model checkpoints are stored in the `models/` folder.
- Visualizations include:
  - Actual vs. Predicted drought severity
  - Feature importance (XGBoost)
  - Loss curves (LSTM)

---

## ✅ Dependencies

Make sure you have Python 3.8+ installed. Key libraries include:

- `pandas`, `numpy`, `scikit-learn`
- `xgboost`
- `tensorflow` / `keras`
- `matplotlib`, `seaborn`
- `jupyter`

Full list in `requirements.txt`.

---

## 📌 To-Do (Enhancement Ideas)

- [ ] Add CLI arguments for training scripts (model choice, epochs, data path)
- [ ] Hyperparameter tuning with Optuna or GridSearchCV
- [ ] Add a Streamlit or Flask interface for live demo
- [ ] Evaluate performance on unseen test datasets

---

## 📬 Contact

Made with 💻 by [Vansh Arora](https://github.com/vansharora21) and [Somya Upadhyay](https://github.com/Somyaaaaa23)
