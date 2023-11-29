from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import label_binarize
import warnings

# Ignorar todas las advertencias de FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------------
# Leer los datasets
df1 = pd.read_csv('bioquimicas.csv', delimiter=';', skipinitialspace=True)
df2 = pd.read_csv('ensaantro2022_entrega_w.csv', delimiter=';', skipinitialspace=True)
df3 = pd.read_csv('ensafisica2022_adultos_entrega_w.csv', delimiter=';', skipinitialspace=True)
df4 = pd.read_csv('ensadul2022_entrega_w.csv', delimiter=';', skipinitialspace=True)

#-----------------------------------------------------------------------------------
# Obtener folios únicos para los conjuntos de datos por separado
folios_unicos_df1 = len(df1['FOLIO_INT'].unique())
folios_unicos_df2 = len(df2['FOLIO_INT'].unique())
folios_unicos_df3 = len(df3['FOLIO_INT'].unique())
folios_unicos_df4 = len(df4['FOLIO_INT'].unique())
#-----------------------------------------------------------------------------------

# Mostrar la cantidad de folios únicos por conjunto de datos
print("Cantidad de folios únicos en el primer Dataset:", folios_unicos_df1)
print("Cantidad de folios únicos en el segundo Dataset:", folios_unicos_df2)
print("Cantidad de folios únicos en el tercer Dataset:", folios_unicos_df3)
print("Cantidad de folios únicos en el cuarta Dataset:", folios_unicos_df4)

#-----------------------------------------------------------------------------------

# Combinar los cinco conjuntos de datos en base a la columna FOLIO_INT
merged_df = pd.merge(df1, df2, on='FOLIO_INT', how='inner', suffixes=('_df1', '_df2'))
merged_df = pd.merge(merged_df, df3, on='FOLIO_INT', how='inner', suffixes=('', '_df3'))
merged_df = pd.merge(merged_df, df4, on='FOLIO_INT', how='inner', suffixes=('', '_df4'))
print(merged_df)

#-----------------------------------------------------------------------------------

df1_selected = merged_df[['a0401', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03',]]
df1_selected2 = merged_df[[ 'a0301', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']]

#-----------------------------------------------------------------------------------

# Mostrar el nuevo DataFrame resultante
print(df1_selected)
print(df1_selected2)

#-----------------------------------------------------------------------------------

cantidadnull = df1_selected2.isnull().sum()
print("\n Se encontraron valores nulos en el DataFrame. \n", cantidadnull)
dfFinalNull2 = df1_selected2.dropna()
print("DataFrame después de dropna:")
print(dfFinalNull2)

#-----------------------------------------------------------------------------------

cantidadnull = df1_selected.isnull().sum()
print("\n Se encontraron valores nulos en el DataFrame. \n", cantidadnull)
dfFinalNull = df1_selected.dropna()
print("DataFrame después de dropna:")
print(dfFinalNull)

#-----------------------------------------------------------------------------------

# Filtrar las filas donde 'a0401' no es igual a 2
dfIpertencion = dfFinalNull[dfFinalNull['a0401'] != 2].copy()
# Mostrar el conjunto de datos después del filtro
print("Dataset Nuevo después de filtrar el valor 2 en 'a0401':")
print(dfIpertencion)

#-----------------------------------------------------------------------------------

dfDiabetes = dfFinalNull2[dfFinalNull2['a0301'] != 2].copy()
# Mostrar el conjunto de datos después del filtro
print("Dataset Nuevo después de filtrar el valor 2 en 'a0301':")
print(dfDiabetes)

#-----------------------------------------------------------------------------------

dfIpertencion.loc[:, 'a0401'] = pd.to_numeric(dfIpertencion['a0401'], errors='coerce')
valor_condicion = 1
dfIpertencion['a0401'] = (dfIpertencion['a0401'] == valor_condicion).astype(int)
print("Dataset Nuevo después de categorizar el valor 3 en 'a0401':")
print(dfIpertencion)

#-----------------------------------------------------------------------------------

dfDiabetes.loc[:, 'a0301'] = pd.to_numeric(dfDiabetes['a0301'], errors='coerce')
valor_condicion = 1
dfDiabetes['a0301'] = (dfDiabetes['a0301'] == valor_condicion).astype(int)
print("Dataset Nuevo después de categorizar el valor 3 en 'a0301':")
print(dfDiabetes)

#-----------------------------------------------------------------------------------

# Reemplazar comas por puntos en las columnas relevantes
columns_to_replace_commas = ['a0401', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']
dfIpertencion[columns_to_replace_commas] = dfIpertencion[columns_to_replace_commas].replace({',': '.'}, regex=True)
dfIpertencion[columns_to_replace_commas] = dfIpertencion[columns_to_replace_commas].astype(float)
dfFloat = dfIpertencion

#-----------------------------------------------------------------------------------

# Reemplazar comas por puntos en las columnas relevantes
columns_to_replace_commas = ['a0301', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']
dfDiabetes[columns_to_replace_commas] = dfDiabetes[columns_to_replace_commas].replace({',': '.'}, regex=True)
dfDiabetes[columns_to_replace_commas] = dfDiabetes[columns_to_replace_commas].astype(float)
dfFloat2 = dfDiabetes

#-----------------------------------------------------------------------------------

# Calculamos la matriz de correlacion
dfCorre = dfFloat.corr(method="pearson")
# Imprimir la matriz de correlación
print("\nMatriz de correlación Hipertencion:")
print(dfCorre)

#-----------------------------------------------------------------------------------

# Calculamos la matriz de correlacion
dfCorre = dfFloat2.corr(method="pearson")
# Imprimir la matriz de correlación
print("\nMatriz de correlación Diabetes:")
print(dfCorre)

#-----------------------------------------------------------------------------------

XHiper = dfFloat[['h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']].values
yHiper = dfFloat['a0401'].values

#-----------------------------------------------------------------------------------


XDiabe = dfFloat2[['h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']].values
yDiabe = dfFloat2['a0301'].values

#-----------------------------------------------------------------------------------

# Normalizar los datos para hipertensión
scaler_hiper = StandardScaler()
XHiper_normalized = scaler_hiper.fit_transform(XHiper)

# Normalizar los datos para diabetes
scaler_diabe = StandardScaler()
XDiabe_normalized = scaler_diabe.fit_transform(XDiabe)

#-----------------------------------------------------------------------------------

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_hiper, X_test_hiper, y_train_hiper, y_test_hiper = train_test_split(XHiper_normalized, yHiper, test_size=0.2, random_state=42)

#-----------------------------------------------------------------------------------

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train_Diabe, X_test_Diabe, y_train_Diabe, y_test_Diabe = train_test_split(XDiabe_normalized, yDiabe, test_size=0.2, random_state=42)

#-----------------------------------------------------------------------------------
""" 
#Almacena la suma de los cuadrados de las distancias para cada número de clusters
sum_of_squared_distances = []
#Prueba con un número de clusters en un rango específico
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(XHiper,yHiper)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Suma de cuadrados de las distancias')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(XHiper)

plt.scatter(XHiper[:, 0], XHiper[:, 1], c=kmeans_labels, cmap='viridis', 
            marker='o', edgecolor='k', s=50)
plt.title("Clusterización con K-Means")
plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
plt.show()

#-----------------------------------------------------------------------------------

#Almacena la suma de los cuadrados de las distancias para cada número de clusters
sum_of_squared_distances = []
#Prueba con un número de clusters en un rango específico
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(XDiabe,yDiabe)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Suma de cuadrados de las distancias')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(XDiabe)

plt.scatter(XDiabe[:, 0], XDiabe[:, 1], c=kmeans_labels, cmap='viridis', 
            marker='o', edgecolor='k', s=50)
plt.title("Clusterización con K-Means")
plt.xlabel("Caracteristica 1")
plt.ylabel("Caracteristica 2")
plt.show()
 """
#-----------------------------------------------------------------------------------

logreg_hiper = LogisticRegression(max_iter=5000)
logreg_hiper.fit(X_train_hiper, y_train_hiper)

# Realizar predicciones en el conjunto de prueba
y_pred_hiper = logreg_hiper.predict(X_test_hiper)
y_prob_hiper = logreg_hiper.predict_proba(X_test_hiper)[:, 1]  # Probabilidad de clase positiva
# Evaluar el rendimiento del modelo
accuracy_hiper = accuracy_score(y_test_hiper, y_pred_hiper)
precision_hiper = precision_score(y_test_hiper, y_pred_hiper)
recall_hiper = recall_score(y_test_hiper, y_pred_hiper)
roc_auc_hiper = roc_auc_score(y_test_hiper, y_pred_hiper)

print("Rendimiento del modelo de regresión logística para hipertensión:")
print("Accuracy:", accuracy_hiper)
print("Precision:", precision_hiper)
print("Recall:", recall_hiper)
print("ROC AUC:", roc_auc_hiper)
# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_hiper,y_prob_hiper)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
#-----------------------------------------------------------------------------------

logreg_diabe = LogisticRegression(max_iter=5000)
logreg_diabe.fit(X_train_Diabe, y_train_Diabe)

# Realizar predicciones en el conjunto de prueba
y_pred_Diabe = logreg_diabe.predict(X_test_Diabe)
y_prob_Diabe = logreg_diabe.predict_proba(X_test_Diabe)[:, 1]  # Probabilidad de clase positiva
# Evaluar el rendimiento del modelo
accuracy_diabe = accuracy_score(y_test_Diabe, y_pred_Diabe)
precision_diabe = precision_score(y_test_Diabe, y_pred_Diabe)
recall_diabe = recall_score(y_test_Diabe, y_pred_Diabe)
roc_auc_diabe = roc_auc_score(y_test_Diabe, y_pred_Diabe)

print("Rendimiento del modelo de regresión logística para diabetes:")
print("Accuracy:", accuracy_diabe)
print("Precision:", precision_diabe)
print("Recall:", recall_diabe)
print("ROC AUC:", roc_auc_diabe)
# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_Diabe,y_prob_Diabe)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------
# Escalar los datos para mejorar el rendimiento de la red neuronal
scaler_hiper = StandardScaler()
X_train_scaled_hiper = scaler_hiper.fit_transform(X_train_hiper)
X_test_scaled_hiper = scaler_hiper.transform(X_test_hiper)

# Crear y entrenar la red neuronal Adaline para hipertensión
adaline_hiper = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
adaline_hiper.fit(X_train_scaled_hiper, y_train_hiper)

# Realizar predicciones en el conjunto de prueba
y_pred_hiper = adaline_hiper.predict(X_test_scaled_hiper)

# Evaluar el rendimiento del modelo
mse_hiper = mean_squared_error(y_test_hiper, y_pred_hiper)
r2_hiper = r2_score(y_test_hiper, y_pred_hiper)

print("Rendimiento de la red neuronal Adaline para hipertensión:")
print("Mean Squared Error:", mse_hiper)
print("R^2 Score:", r2_hiper)
umbral = 0.5
y_pred_hiper_binario = (y_pred_hiper > umbral).astype(int)

precision_hiper = precision_score(y_test_hiper, y_pred_hiper_binario)
recall_hiper = recall_score(y_test_hiper, y_pred_hiper_binario)
roc_auc_hiper = roc_auc_score(y_test_hiper, y_pred_hiper_binario)

print("Métricas de clasificación para Adaline (hipertensión):")
print("Precision:", precision_hiper)
print("Recall:", recall_hiper)
print("ROC AUC:", roc_auc_hiper)

#-----------------------------------------------------------------------------------

# Escalar los datos para mejorar el rendimiento de la red neuronal
scaler_diabe = StandardScaler()
X_train_scaled_diabe = scaler_diabe.fit_transform(X_train_Diabe)
X_test_scaled_diabe = scaler_diabe.transform(X_test_Diabe)
# Crear y entrenar la red neuronal Adaline para diabetes
adaline_diabe = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
adaline_diabe.fit(X_train_scaled_diabe, y_train_Diabe)

# Realizar predicciones en el conjunto de prueba
y_pred_diabe = adaline_diabe.predict(X_test_scaled_diabe)

# Evaluar el rendimiento del modelo
mse_diabe = mean_squared_error(y_test_Diabe, y_pred_diabe)
r2_diabe = r2_score(y_test_Diabe, y_pred_diabe)

print("Rendimiento de la red neuronal Adaline para diabetes:")
print("Mean Squared Error:", mse_diabe)
print("R^2 Score:", r2_diabe)

# #-----------------------------------------------------------------------------------

# # Crear y entrenar el modelo KNN para hipertensión
# knn_hiper = KNeighborsRegressor(n_neighbors=5)

# knn_hiper.fit(X_train_hiper, y_train_hiper)

# # Realizar predicciones en el conjunto de prueba
# y_pred_hiper = knn_hiper.predict(X_test_hiper)
# # Convert las etiquetas de clase a formato binario
# y_true_binary = label_binarize(y_test_Diabe, classes=[0, 1]).ravel()

# # Visualizar la curva ROC

# # Evaluar el rendimiento del modelo
# mse_hiper = mean_squared_error(y_test_hiper, y_pred_hiper)
# r2_hiper = r2_score(y_test_hiper, y_pred_hiper)

# print("Rendimiento del modelo KNN para hipertensión:")
# print("Mean Squared Error:", mse_hiper)
# print("R^2 Score:", r2_hiper)


# #-----------------------------------------------------------------------------------

# # Crear y entrenar el modelo KNN para diabetes
# knn_diabe = KNeighborsRegressor(n_neighbors=5)
# knn_diabe.fit(X_train_Diabe, y_train_Diabe)

# # Realizar predicciones en el conjunto de prueba
# y_pred_diabe = knn_diabe.predict(X_test_Diabe)

# # Evaluar el rendimiento del modelo
# mse_diabe = mean_squared_error(y_test_Diabe, y_pred_diabe)
# r2_diabe = r2_score(y_test_Diabe, y_pred_diabe)

# print("Rendimiento del modelo KNN para diabetes:")
# print("Mean Squared Error:", mse_diabe)
# print("R^2 Score:", r2_diabe)

# #-----------------------------------------------------------------------------------
