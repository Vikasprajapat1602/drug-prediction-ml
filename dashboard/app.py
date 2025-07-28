# dashboard/app.py

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# ----------------------------------
# 1. Define Model Architecture
# ----------------------------------
class DrugModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DrugModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------
# 2. Streamlit UI Start
# ----------------------------------
st.title("💊 Drug Prediction Dashboard (Deep Learning)")

st.sidebar.markdown("<h4 style='color:white; font-size:20px; font-weight:bold; margin-bottom:6px;'>🔍 Choose Prediction Task</h4>", unsafe_allow_html=True)
task = st.sidebar.selectbox("", ["Drug Type", "Cannabis Use"])


# ----------------------------------
# 3. Drug Type Prediction
# ----------------------------------
if task == "Drug Type":
    st.markdown("Predict drug based on patient's health parameters")

    df = pd.read_csv(r"D:\Internship_assignment\drug_discovery_ml\data\drugs_data.csv")
    le_sex, le_bp, le_chol, le_drug = (
        LabelEncoder(),
        LabelEncoder(),
        LabelEncoder(),
        LabelEncoder(),
    )

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["BP"] = le_bp.fit_transform(df["BP"])
    df["Cholesterol"] = le_chol.fit_transform(df["Cholesterol"])
    df["Drug"] = le_drug.fit_transform(df["Drug"])

    X = df.drop("Drug", axis=1)
    y = df["Drug"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DrugModel(input_size=X.shape[1], output_size=len(y.unique()))
    model.load_state_dict(
        torch.load(r"D:\Internship_assignment\drug_discovery_ml\models\drug_model.pth")
    )
    model.eval()

    # ---- Sidebar inputs ----
    st.sidebar.header("Enter Patient Details:")
    age = st.sidebar.slider("Age", 10, 80, 30, key="age_drug")
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"), key="sex_drug")
    bp = st.sidebar.selectbox(
        "Blood Pressure", ("Low", "Normal", "High"), key="bp_drug"
    )
    chol = st.sidebar.selectbox("Cholesterol", ("Normal", "High"), key="chol_drug")
    na_to_k = st.sidebar.slider("Na_to_K Ratio", 5.0, 40.0, 15.0, key="na_to_k_drug")

    # Value mappings
    sex_map = {"Male": "M", "Female": "F"}
    bp_map = {"Low": "LOW", "Normal": "NORMAL", "High": "HIGH"}
    chol_map = {"Normal": "NORMAL", "High": "HIGH"}

    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Sex": le_sex.transform([sex_map[sex]]),
            "BP": le_bp.transform([bp_map[bp]]),
            "Cholesterol": le_chol.transform([chol_map[chol]]),
            "Na_to_K": [na_to_k],
        }
    )

    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        pred_label = le_drug.inverse_transform([pred_class])[0]

    st.success(f"✅ Predicted Drug: **{pred_label}**")

    # Dataset preview + Accuracy
    with st.expander("📄 View Sample Dataset"):
        st.write(df.head())

    with torch.no_grad():
        y_pred = torch.argmax(model(torch.tensor(X_scaled, dtype=torch.float32)), dim=1)
        acc = accuracy_score(y, y_pred)
    st.write(f"📈 Model Accuracy: **{acc * 100:.2f}%**")

    cm = confusion_matrix(y, y_pred)
    st.subheader("🔍 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ----------------------------------
# 4. Cannabis Use Prediction
# ----------------------------------
elif task == "Cannabis Use":
    st.markdown("Predict cannabis usage from psychological traits")

    df = pd.read_csv(
        r"D:\Internship_assignment\drug_discovery_ml\data\Drug_Consumption.csv"
    )
    features = ["Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS"]
    df = df[features + ["Cannabis"]].dropna()

    # Binarize target: 1 = user, 0 = non-user
    df["Cannabis"] = df["Cannabis"].apply(lambda x: 1 if x not in ["CL0", "CL1"] else 0)

    X = df.drop("Cannabis", axis=1)
    y = df["Cannabis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DrugModel(input_size=X.shape[1], output_size=2)
    model.load_state_dict(
        torch.load(
            r"D:\Internship_assignment\drug_discovery_ml\models\cannabis_model.pth"
        )
    )
    model.eval()

    # ---- Sidebar inputs ----
    st.sidebar.header("Enter Personality Traits:")
    nscore = st.sidebar.slider(
        "Neuroticism (Nscore)", 0.0, 1.0, 0.5, key="nscore_cannabis"
    )
    escore = st.sidebar.slider(
        "Extraversion (Escore)", 0.0, 1.0, 0.5, key="escore_cannabis"
    )
    oscore = st.sidebar.slider(
        "Openness (Oscore)", 0.0, 1.0, 0.5, key="oscore_cannabis"
    )
    ascore = st.sidebar.slider(
        "Agreeableness (Ascore)", 0.0, 1.0, 0.5, key="ascore_cannabis"
    )
    cscore = st.sidebar.slider(
        "Conscientiousness (Cscore)", 0.0, 1.0, 0.5, key="cscore_cannabis"
    )
    impulsive = st.sidebar.slider(
        "Impulsiveness", 0.0, 1.0, 0.5, key="impulsive_cannabis"
    )
    ss = st.sidebar.slider("Sensation Seeking", 0.0, 1.0, 0.5, key="ss_cannabis")

    input_df = pd.DataFrame(
        {
            "Nscore": [nscore],
            "Escore": [escore],
            "Oscore": [oscore],
            "Ascore": [ascore],
            "Cscore": [cscore],
            "Impulsive": [impulsive],
            "SS": [ss],
        }
    )

    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        result = "Cannabis User" if pred_class == 1 else "Non-user"

    st.success(f"🧠 Prediction: **{result}**")

    # Dataset preview + Accuracy
    with st.expander("📄 View Sample Dataset"):
        st.write(df.head())

    with torch.no_grad():
        y_pred = torch.argmax(model(torch.tensor(X_scaled, dtype=torch.float32)), dim=1)
        acc = accuracy_score(y, y_pred)
    st.write(f"📈 Model Accuracy: **{acc * 100:.2f}%**")

    cm = confusion_matrix(y, y_pred)
    st.subheader("🧪 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="g", cmap="Purples", ax=ax)
    st.pyplot(fig)
