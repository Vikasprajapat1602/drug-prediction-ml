Here is the **complete `README.md` file** content in one go — just **copy and paste** this into a new file named `README.md` inside your project folder:

---

```markdown
# 💊 Drug Prediction Dashboard (Deep Learning)

This project is a **Streamlit-based machine learning dashboard** that predicts:

1. **Drug type** based on patient health parameters using a neural network.
2. **Cannabis usage** based on personality traits using a second neural network.

It uses **PyTorch**, **scikit-learn**, and **Streamlit** for building, training, and deploying interactive models.

---

## 📁 Folder Structure

```

drug\_discovery\_ml/
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── data/
│   ├── drugs\_data.csv        # Dataset 1: Drug type prediction
│   └── Drug\_Consumption.csv  # Dataset 2: Cannabis usage
│
├── models/
│   ├── drug\_model.pth        # Trained model for Drug type
│   └── cannabis\_model.pth    # Trained model for Cannabis prediction
│
├── requirements.txt          # Dependencies
└── README.md                 # This file

````

---

## 🧠 Features

### 1. Drug Type Prediction
- **Inputs**: Age, Sex, Blood Pressure, Cholesterol, Sodium-to-Potassium ratio.
- **Output**: Predicted **drug type** based on the patient's health parameters.

### 2. Cannabis Use Prediction
- **Inputs**: Psychological traits like Neuroticism, Openness, Agreeableness, etc.
- **Output**: Likelihood of **Cannabis usage** (User or Non-user).

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Vikasprajapat1602/drug-prediction-ml.git
cd drug-prediction-ml
````

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
cd dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📊 Screenshots

> *(Add your own screenshots here if available for better presentation)*
> Example:

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Screenshot)

---

## 🧾 Technologies Used

* Python
* PyTorch
* Scikit-learn
* Pandas
* Seaborn
* Matplotlib
* Streamlit

---

## 📚 Datasets Used

1. **drugs\_data.csv**

   * Source: Simulated patient data
   * Use: Predicting drug type based on health indicators

2. **Drug\_Consumption.csv**

   * Source: Personality trait data + drug usage
   * Use: Predicting cannabis user vs non-user

---

## 🙋‍♂️ Author

**Vikas Prajapat**
📌 GitHub: [Vikasprajapat1602](https://github.com/Vikasprajapat1602)

---

## 📌 License

This project is open source and available under the [MIT License](LICENSE).

````

---

Let me know if you want:
- A Hindi explanation version
- A GitHub banner
- A contribution or credits section added

Once added, don’t forget to commit:

```bash
git add README.md
git commit -m "Added full README.md"
git push
````
