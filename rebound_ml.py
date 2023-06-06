# libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

# importing the data.
calibracion_df = pd.read_csv("Calibración esclerómetro.csv")
calibracion_df.rename(columns = {"As. Obj.": "As_Obj"}, inplace= True)
calibracion_df.rename(columns = {"TMN": "Piedra"}, inplace= True)

# data preprocessing.
calibracion_df["Piedra"] = calibracion_df["Piedra"].astype("string")
calibracion_df["Especificada"] = calibracion_df["Especificada"].astype("string")
calibracion_df["Paston"] = calibracion_df["Paston"].astype("string")
calibracion_df["Cemento"] = calibracion_df["Cemento"].astype("string")

output = calibracion_df["Rotura"]
features = calibracion_df[["Piedra", "Especificada", "Tenor", "Paston", "Cemento","Edad", "Rebote"]]

# one-hot encoding.
features = pd.get_dummies(features)
features.rename(columns = {"Cemento_Loma Negra":"Cemento_Loma_Negra"}, inplace=True)
print(features.columns)
# separating the data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = .2)

# Training the model
# n_estimators=1000, learning_rate=0.01, max_depth = 5, min_child_weight = 1, gamma = 10, subsample = 0.5, objective= "reg:squarederror"
regressor = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth = 5, min_child_weight = 1,
                         gamma = 10, subsample = 0.5, objective= "reg:squarederror")
regressor.fit(x_train, y_train,
              early_stopping_rounds = 4,
              eval_set = [(x_test, y_test)],
              verbose = False)

y_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

r_2_train = r2_score(y_train, y_pred_train)
r_2_test = r2_score(y_test, y_pred_test)

print(r_2_train)
print(r_2_test)

explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(features)

fig, ax = plt.subplots()
ax = shap.summary_plot(shap_values, features.columns ,plot_type = "bar")
plt.show()
fig.savefig("Feature_Shap_Values.png")

from yellowbrick.regressor import ResidualsPlot

fig2, ax = plt.subplots()
ax = visualizer = ResidualsPlot(regressor)
visualizer.fit(x_train, y_train)  # Fit the training data to the visualizer
visualizer.score(x_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure
fig2.savefig("Residuals.png")

xgb_pickle = open("XGB_model.pickle", "wb")
pickle.dump(regressor, xgb_pickle)
xgb_pickle.close()

print(regressor.feature_names_in_)