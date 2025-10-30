import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 💖 Configuración de la página
st.set_page_config(
    page_title="Titanic Data Explorer 💖🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 💅 CSS Barbie Theme
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #FFD1DC, #FFB6C1, #FFE4E1);
            color: #C71585;
            font-family: 'Poppins', sans-serif;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Pacifico', cursive;
            color: #E0218A;
            text-shadow: 0px 0px 6px #FFD1DC;
        }

        .stMetric {
            background-color: rgba(255, 182, 193, 0.6);
            border: 2px solid #FFD700;
            border-radius: 20px;
            padding: 10px;
            color: #C71585;
        }

        .stButton>button {
            background: linear-gradient(to right, #FF69B4, #E0218A);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 8px 20px;
            font-size: 16px;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background: linear-gradient(to right, #FFB6C1, #FF69B4);
            box-shadow: 0 0 15px #FFD700;
            color: #fff;
            transform: scale(1.05);
        }

        .stSidebar {
            background: #FFB6C1;
            color: #C71585;
            border-right: 3px solid #FFD700;
        }

        .stTabs [role="tab"] {
            background: #FFD1DC;
            color: #E0218A;
            border-radius: 10px;
            font-weight: bold;
        }

        .stTabs [role="tab"][aria-selected="true"] {
            background: #E0218A;
            color: white;
            box-shadow: 0px 0px 10px #FFD700;
        }

        div[data-testid="stMetricValue"] {
            color: #E0218A;
            font-weight: bold;
        }

        a, a:visited {
            color: #E0218A;
        }

        a:hover {
            color: #FFD700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🌸 Encabezado principal
st.markdown("<h1 style='text-align: center;'>💖 Titanic Data Explorer 🚢✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analiza los datos del Titanic </p>", unsafe_allow_html=True)

# 🎀 Sidebar
st.sidebar.title("💖 Configuración Dashboard")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV del Titanic", type=["csv"])

barbie_palette = ['#E0218A', '#FF69B4', '#FFD1DC', '#FFB6C1', '#FFD700']
barbie_cmap = sns.color_palette(barbie_palette, as_cmap=True)
barbie_cmap = LinearSegmentedColormap.from_list("barbie", barbie_palette)

# Si no se ha subido archivo
if uploaded_file is None:
    st.info("👑 Sube el archivo `titanic.csv` para comenzar tu análisis  ✨")
    st.stop()

# 📂 Cargar datos
titanic = pd.read_csv(uploaded_file)
st.success("🌟 Datos cargados correctamente. ¡Vamos a explorar!")

# 🧹 Limpieza básica
titanic = titanic.dropna(subset=['Age', 'Fare', 'Sex', 'Pclass', 'Survived'])
titanic['Sex'] = titanic['Sex'].map({'male': 'Hombre', 'female': 'Mujer'})

# 📊 Métricas principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pasajeros", len(titanic))
col2.metric("Supervivientes", titanic['Survived'].sum())
col3.metric("Edad Media", f"{titanic['Age'].mean():.1f}")
col4.metric("Tarifa Media", f"{titanic['Fare'].mean():.1f}")

# ✨ Tabs principales
tabs = st.tabs(["📈 Dashboard", "👩‍🎓 Distribuciones", " Machine Learning"])

# --- TAB 1: Dashboard general
with tabs[0]:
    st.subheader("💖 Distribución de Supervivencia por Sexo y Clase")
    fig = px.histogram(
        titanic, x='Pclass', color='Sex', barmode='group',
        facet_col='Survived', category_orders={"Survived": [0, 1]},
        color_discrete_sequence=barbie_palette,
        title="Distribución de pasajeros por clase, sexo y supervivencia"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎀 Distribución de edades")
    fig2 = px.histogram(
        titanic, x='Age', nbins=20, color='Sex',
        color_discrete_sequence=barbie_palette,
        title="Distribución de edades por sexo"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- TAB 2: Distribuciones
with tabs[1]:
    st.subheader("💄 Relación Edad vs Tarifa")
    fig3 = px.scatter(
        titanic, x='Age', y='Fare', color='Sex',
        color_discrete_sequence=barbie_palette,
        hover_data=['Pclass', 'Survived'],
        title="Relación entre Edad y Tarifa"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🌸 Mapa de calor de correlaciones")
    corr = titanic[['Survived', 'Age', 'Fare', 'Pclass']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="pink", ax=ax)
    st.pyplot(fig)

# --- TAB 3: Machine Learning
with tabs[2]:
    st.subheader("🤖  ML Predictor")
    titanic_ml = titanic.copy()
    titanic_ml['Sex'] = titanic_ml['Sex'].map({'Hombre': 0, 'Mujer': 1})
    X = titanic_ml[['Pclass', 'Sex', 'Age', 'Fare']]
    y = titanic_ml['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Precisión del modelo", f"{acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax_cm, cmap=barbie_cmap, colorbar=False)
    ax_cm.set_title("💖 Matriz de Confusión 💖", fontsize=16, color="#FF69B4")
    ax_cm.set_xlabel("Predicción", fontsize=12, color="#D63384")
    ax_cm.set_ylabel("Valor Real", fontsize=12, color="#D63384")
    plt.setp(ax_cm.get_xticklabels(), color="#C71585")
    plt.setp(ax_cm.get_yticklabels(), color="#C71585")
    st.pyplot(fig_cm)

    st.subheader("💕 Predicción interactiva")
    pclass = st.selectbox("Clase", [1, 2, 3])
    sex = st.selectbox("Sexo", ["Hombre", "Mujer"])
    age = st.slider("Edad", 1, 80, 25)
    fare = st.slider("Tarifa", 0, 500, 50)

    sex_val = 0 if sex == "Hombre" else 1
    prediction = model.predict([[pclass, sex_val, age, fare]])[0]

    if prediction == 1:
        st.success("🎀 ¡Sobrevivirías al Titanic! 💅")
    else:
        st.error("😢 Lo siento... ni el glamour te salvaría en este viaje.")
