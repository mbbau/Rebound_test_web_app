import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.title("Predicción de Resistencia mediante Esclerometría y Machine Learning")

st.write("Esta web app suma más variables del hormigón a los resultados de la Esclerometría" 
         "para obtener una aproximación de la resistencia más precisa que si se utilizara unicamente"
         "el rebote y una regresión lineal a partir de este.")

st.write("El estudio que dió origen a esta web app puede encontrarse en el siguiente [repositorio](https://github.com/mbbau/About-rebound-test-and-its-models-of-prediction)")

# Selección de nuevas variables dadas por el usuario. 
# La lista de variables a entregar por el usuario son: 
# ["TMN", "As. Obj.","Especificada", "Tenor", "Paston", "Cemento","Edad", "Rebote"]
st.sidebar.subheader("Selección de variables")

Rebote = st.sidebar.number_input("Promedio rebote esclerometría", min_value=20, max_value=50)
Edad = st.sidebar.number_input("Edad del hormigón", min_value=3)
Cemento = st.sidebar.selectbox("Cemento", options = ["Loma Negra", "Holcim", "Avellaneda"])
Paston = st.sidebar.selectbox("¿Es un pastón de laboratorio?", options = ["Si", "No"])
Tenor = st.sidebar.number_input("Tenor cemento teórico", min_value = 300, max_value = 500)
Especificada = st.sidebar.selectbox("Resistencia Especificada", options = ["13", "17", "21", "25", "30", "35"])
Piedra = st.sidebar.selectbox("Tamaño Máximo Nominal", options = ["12", "19", "25", "30"])

st.subheader("Variables Nuevas")
st.write("Las variables ingresadas por el usuario son:")
st.markdown("* *Rebote:* {}".format([Rebote]))
st.markdown("* *Edad:* {}".format([Edad]))
st.markdown("* *Cemento:* {}".format([Cemento]))
st.markdown("* *¿La muestra proviene de un pastón?:* {}".format([Paston]))
st.markdown("* *Tenor Cemento Teórico:* {}".format([Tenor]))
st.markdown("* *Resistencia Especificada* {}".format([Especificada])) 

Piedra_12, Piedra_19, Piedra_25, Piedra_30 = 0, 0, 0, 0
if Piedra == "12":
    Piedra_12 = 1
elif Piedra == "19":
    Piedra_19 = 1
elif Piedra == "25":
    Piedra_25 = 1
elif Piedra == "30":
    Piedra_30 = 1

Especificada_13, Especificada_17, Especificada_21, Especificada_25, Especificada_30, Especificada_35 = 0, 0, 0, 0, 0, 0
if Especificada == "13":
    Especificada_132 = 1
elif Especificada == "17":
    Especificada_17 = 1
elif Especificada == "21":
    Especificada_21 = 1
elif Especificada == "25":
    Especificada_25 = 1
elif Especificada == "30":
    Especificada_30 = 1
elif Especificada == "35":
    Especificada_35 = 1

Paston_No , Paston_Si = 0, 0
if Paston == "Si":
    Paston_Si = 1
else:
    Paston_No = 1

Cemento_Avellaneda, Cemento_Holcim, Cemento_Loma_Negra = 0, 0, 0
if Cemento == "Avellaneda":
    Cemento_Avellaneda = 1
elif Cemento == "Holcim":
    Cemento_Holcim = 1
elif Cemento == "Loma Negra":
    Cemento_Loma_Negra = 1

xgb_pickle = open("XGB_model.pickle", "rb")
regressor = pickle.load(xgb_pickle)
xgb_pickle.close()

x_nuevo = np.array([[Tenor, Edad, Rebote, Piedra_12, Piedra_19, Piedra_25, Piedra_30, 
                     Especificada_13, Especificada_17, Especificada_21, Especificada_25, 
                     Especificada_30, Especificada_35, Paston_No, Paston_Si, 
                     Cemento_Avellaneda, Cemento_Holcim, Cemento_Loma_Negra]])

y_predict = regressor.predict(x_nuevo)

st.subheader("Predicción de Resistencia")

st.write("La resistencia en Megapascales estimada del hormigón analizado es: {}".format(y_predict))

# Gráficos que caracterizan el nuevo modelo
st.subheader("Importancia de los parámetros basados en Shapley Values")
st.write("En el siguiente gráfico, pueden observarse la importancia de las diferentes variables utilizadas para" 
         "realizar la predicción, basado en el cálculo de SHAP Values. Como se puede apreciar, existe una fuerte"
         "Correlación entre la resistencia y el rebote, pero además, la presencia de las variables extra también arroja"
         "información sobre la resistencia del hormigón. SHAP (Shapley Additive exPlanations) es un enfoque basado"
         "en la teoría de juegos que busca explicar los resultado obtenidos por los modelo de machine learning")

st.image("Feature_Shap_Values.png")

st.markdown("Para más información sobre los Shap Values dirigirse a [SHAP](https://shap.readthedocs.io/en/latest/index.html)")

st.subheader("Distribución de valores residuales")
st.write("A continuación pueden observarse las predicciones obtenidas por el modelo junto con la distribución de" 
         "residuales obtenida por el mismo. Este gráfico permite establecer la distribución de los errores"
         "obtenidos por la estimación realizada de la resistencia.")
st.image("Residuals.png")