# 💊 Drug Prediction Dashboard (Deep Learning)

An interactive Streamlit dashboard using deep learning models to predict:

* Drug type based on patient health parameters.
* Cannabis usage based on psychological traits.

---

## 🧠 Project Highlights

### 1. Drug Type Prediction

* 🔹 **Input**: Age, Sex, Blood Pressure, Cholesterol, Na\_to\_K ratio
* 🔹 **Output**: Recommended drug category (A–E)
* 🔹 **Model**: Neural Network (PyTorch)

### 2. Cannabis Use Prediction

* 🔹 **Input**: Personality scores (Nscore, Escore, etc.)
* 🔹 **Output**: User / Non-user classification
* 🔹 **Model**: Binary classifier (PyTorch)

---

## 📁 Folder Structure

```
drug_discovery_ml/
├── dashboard/
│   └── app.py
├── data/
│   ├── drugs_data.csv
│   └── Drug_Consumption.csv
├── models/
│   ├── drug_model.pth
│   └── cannabis_model.pth
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Vikasprajapat1602/drug-prediction-ml.git
cd drug-prediction-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
cd dashboard
streamlit run app.py
```

---

## 🛠 Technologies Used

* Python, Pandas, Scikit-learn
* PyTorch (Neural Networks)
* Streamlit (UI/UX)
* Seaborn + Matplotlib (Visualization)

---

## 📊 Sample Output

* Drug Prediction: `DrugY`
* Cannabis Classification: `Cannabis User`

---

## 🙋‍♂️ About Me

**Vikas Prajapat**
📍 GitHub: [@Vikasprajapat1602](https://github.com/Vikasprajapat1602)

