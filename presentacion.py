from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_curve
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import label_binarize
import warnings

from sklearn.svm import SVC

# Ignorar todas las advertencias de FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

#-----------------------------------------------------------------------------------
# # Leer los datasets
# df1 = pd.read_csv('bioquimicas.csv', delimiter=';', skipinitialspace=True)
# df2 = pd.read_csv('ensaantro2022_entrega_w.csv', delimiter=';', skipinitialspace=True)
# df3 = pd.read_csv('ensafisica2022_adultos_entrega_w.csv', delimiter=';', skipinitialspace=True)
# df4 = pd.read_csv('ensadul2022_entrega_w.csv', delimiter=';', skipinitialspace=True)
# df5 = pd.read_csv('Alimentos.csv')

# #-----------------------------------------------------------------------------------
# # Obtener folios únicos para los conjuntos de datos por separado
# folios_unicos_df1 = len(df1['FOLIO_INT'].unique())
# folios_unicos_df2 = len(df2['FOLIO_INT'].unique())
# folios_unicos_df3 = len(df3['FOLIO_INT'].unique())
# folios_unicos_df4 = len(df4['FOLIO_INT'].unique())
# folios_unicos_df5 = len(df5['FOLIO_INT'].unique())
# #-----------------------------------------------------------------------------------

# # Mostrar la cantidad de folios únicos por conjunto de datos
# print("Cantidad de folios únicos en el primer Dataset:", folios_unicos_df1)
# print("Cantidad de folios únicos en el segundo Dataset:", folios_unicos_df2)
# print("Cantidad de folios únicos en el tercer Dataset:", folios_unicos_df3)
# print("Cantidad de folios únicos en el cuarta Dataset:", folios_unicos_df4)
# print("Cantidad de folios únicos en el cuarta Dataset:", folios_unicos_df5)

# #-----------------------------------------------------------------------------------

# # Combinar los cinco conjuntos de datos en base a la columna FOLIO_INT
# merged_df = pd.merge(df1, df2, on='FOLIO_INT', how='inner', suffixes=('_df1', '_df2'))
# merged_df = pd.merge(merged_df, df3, on='FOLIO_INT', how='inner', suffixes=('', '_df3'))
# merged_df = pd.merge(merged_df, df4, on='FOLIO_INT', how='inner', suffixes=('', '_df4'))
# merged_df = pd.merge(merged_df, df4, on='FOLIO_INT', how='inner', suffixes=('', '_df5'))
# print(merged_df)

# #-----------------------------------------------------------------------------------

# df1_selected = merged_df[['a0401', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03','Calorias_semana','Proteina_semana','CarboHidratos_semana','Lipidos_semana']]
# df1_selected2 = merged_df[[ 'a0301', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03','Calorias_semana','Proteina_semana','CarboHidratos_semana','Lipidos_semana']]

# #-----------------------------------------------------------------------------------

# # Mostrar el nuevo DataFrame resultante
# print(df1_selected)
# print(df1_selected2)

df1_selected = pd.read_csv('DataFinalHipertencion.csv')
df1_selected2 = pd.read_csv('DataFinalDiabetes.csv')

# #-----------------------------------------------------------------------------------

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
dfIpertencion.to_csv('dfIpertencion.csv', index=False)
#-----------------------------------------------------------------------------------

dfDiabetes = dfFinalNull2[dfFinalNull2['a0301'] != 2].copy()
# Mostrar el conjunto de datos después del filtro
print("Dataset Nuevo después de filtrar el valor 2 en 'a0301':")
print(dfDiabetes)
dfDiabetes.to_csv('dfDiabetes.csv', index=False)
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

#-----------------------------------------------------------------------------------

# Crear un modelo de KNN
knn_hiper = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos según sea necesario

# Entrenar el modelo con los datos de entrenamiento
knn_hiper.fit(X_train_scaled_hiper, y_train_hiper)

# Realizar predicciones en el conjunto de prueba
y_pred_hiper_knn = knn_hiper.predict(X_test_scaled_hiper)
y_prob_hiper_knn = knn_hiper.predict_proba(X_test_scaled_hiper)[:, 1]  # Probabilidad de clase positiva

# Evaluar el rendimiento del modelo
accuracy_hiper_knn = accuracy_score(y_test_hiper, y_pred_hiper_knn)
precision_hiper_knn = precision_score(y_test_hiper, y_pred_hiper_knn)
recall_hiper_knn = recall_score(y_test_hiper, y_pred_hiper_knn)
roc_auc_hiper_knn = roc_auc_score(y_test_hiper, y_pred_hiper_knn)

print("Rendimiento del modelo KNN para hipertensión:")
print("Accuracy:", accuracy_hiper_knn)
print("Precision:", precision_hiper_knn)
print("Recall:", recall_hiper_knn)
print("ROC AUC:", roc_auc_hiper_knn)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_hiper, y_prob_hiper_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para KNN (hipertensión)')
plt.legend(loc='lower right')
plt.show()


#-----------------------------------------------------------------------------------

# Crear y entrenar el modelo KNN para diabetes
knn_diabe = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos según sea necesario
knn_diabe.fit(X_train_scaled_diabe, y_train_Diabe)

# Realizar predicciones y evaluar el rendimiento para diabetes
y_pred_diabe_knn = knn_diabe.predict(X_test_scaled_diabe)
y_prob_diabe_knn = knn_diabe.predict_proba(X_test_scaled_diabe)[:, 1]  # Probabilidad de clase positiva

accuracy_diabe_knn = accuracy_score(y_test_Diabe, y_pred_diabe_knn)
precision_diabe_knn = precision_score(y_test_Diabe, y_pred_diabe_knn)
recall_diabe_knn = recall_score(y_test_Diabe, y_pred_diabe_knn)
roc_auc_diabe_knn = roc_auc_score(y_test_Diabe, y_pred_diabe_knn)

print("Rendimiento del modelo KNN para diabetes:")
print("Accuracy:", accuracy_diabe_knn)
print("Precision:", precision_diabe_knn)
print("Recall:", recall_diabe_knn)
print("ROC AUC:", roc_auc_diabe_knn)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_Diabe, y_prob_diabe_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para KNN (diabetes)')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------
# Crear un modelo de SVM
svm_hiper = SVC(kernel='linear', probability=True)  # Puedes ajustar el kernel según sea necesario

# Entrenar el modelo con los datos de entrenamiento escalados
svm_hiper.fit(X_train_scaled_hiper, y_train_hiper)

# Realizar predicciones en el conjunto de prueba
y_pred_hiper_svm = svm_hiper.predict(X_test_scaled_hiper)
y_prob_hiper_svm = svm_hiper.predict_proba(X_test_scaled_hiper)[:, 1]  # Probabilidad de clase positiva

# Evaluar el rendimiento del modelo
accuracy_hiper_svm = accuracy_score(y_test_hiper, y_pred_hiper_svm)
precision_hiper_svm = precision_score(y_test_hiper, y_pred_hiper_svm)
recall_hiper_svm = recall_score(y_test_hiper, y_pred_hiper_svm)
roc_auc_hiper_svm = roc_auc_score(y_test_hiper, y_pred_hiper_svm)

print("Rendimiento del modelo SVM para hipertensión:")
print("Accuracy:", accuracy_hiper_svm)
print("Precision:", precision_hiper_svm)
print("Recall:", recall_hiper_svm)
print("ROC AUC:", roc_auc_hiper_svm)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_hiper, y_prob_hiper_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para SVM (hipertensión)')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------

# Crear y entrenar el modelo SVM para diabetes
svm_diabe = SVC(kernel='linear', probability=True)  # Puedes ajustar el kernel según sea necesario
svm_diabe.fit(X_train_scaled_diabe, y_train_Diabe)

# Realizar predicciones y evaluar el rendimiento para diabetes
y_pred_diabe_svm = svm_diabe.predict(X_test_scaled_diabe)
y_prob_diabe_svm = svm_diabe.predict_proba(X_test_scaled_diabe)[:, 1]  # Probabilidad de clase positiva

accuracy_diabe_svm = accuracy_score(y_test_Diabe, y_pred_diabe_svm)
precision_diabe_svm = precision_score(y_test_Diabe, y_pred_diabe_svm)
recall_diabe_svm = recall_score(y_test_Diabe, y_pred_diabe_svm)
roc_auc_diabe_svm = roc_auc_score(y_test_Diabe, y_pred_diabe_svm)

print("Rendimiento del modelo SVM para diabetes:")
print("Accuracy:", accuracy_diabe_svm)
print("Precision:", precision_diabe_svm)
print("Recall:", recall_diabe_svm)
print("ROC AUC:", roc_auc_diabe_svm)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_Diabe, y_prob_diabe_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para SVM (diabetes)')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------
# Crear un modelo de Random Forest
random_forest_hiper = RandomForestClassifier(n_estimators=100, random_state=42)  # Puedes ajustar los parámetros según sea necesario

# Entrenar el modelo con los datos de entrenamiento
random_forest_hiper.fit(X_train_scaled_hiper, y_train_hiper)
# Realizar predicciones en el conjunto de prueba
y_pred_hiper_rf = random_forest_hiper.predict(X_test_scaled_hiper)
y_prob_hiper_rf = random_forest_hiper.predict_proba(X_test_scaled_hiper)[:, 1]  # Probabilidad de clase positiva

# Evaluar el rendimiento del modelo
accuracy_hiper_rf = accuracy_score(y_test_hiper, y_pred_hiper_rf)
precision_hiper_rf = precision_score(y_test_hiper, y_pred_hiper_rf)
recall_hiper_rf = recall_score(y_test_hiper, y_pred_hiper_rf)
roc_auc_hiper_rf = roc_auc_score(y_test_hiper, y_pred_hiper_rf)

print("Rendimiento del modelo Random Forest para hipertensión:")
print("Accuracy:", accuracy_hiper_rf)
print("Precision:", precision_hiper_rf)
print("Recall:", recall_hiper_rf)
print("ROC AUC:", roc_auc_hiper_rf)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_hiper, y_prob_hiper_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para Random Forest (hipertensión)')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------

# Crear y entrenar el modelo Random Forest para diabetes
random_forest_diabe = RandomForestClassifier(n_estimators=100, random_state=42)  # Puedes ajustar los parámetros según sea necesario
random_forest_diabe.fit(X_train_scaled_diabe, y_train_Diabe)

# Realizar predicciones y evaluar el rendimiento para diabetes
y_pred_diabe_rf = random_forest_diabe.predict(X_test_scaled_diabe)
y_prob_diabe_rf = random_forest_diabe.predict_proba(X_test_scaled_diabe)[:, 1]  # Probabilidad de clase positiva

accuracy_diabe_rf = accuracy_score(y_test_Diabe, y_pred_diabe_rf)
precision_diabe_rf = precision_score(y_test_Diabe, y_pred_diabe_rf)
recall_diabe_rf = recall_score(y_test_Diabe, y_pred_diabe_rf)
roc_auc_diabe_rf = roc_auc_score(y_test_Diabe, y_pred_diabe_rf)

print("Rendimiento del modelo Random Forest para diabetes:")
print("Accuracy:", accuracy_diabe_rf)
print("Precision:", precision_diabe_rf)
print("Recall:", recall_diabe_rf)
print("ROC AUC:", roc_auc_diabe_rf)

# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test_Diabe, y_prob_diabe_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([1, 1], [1, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC para Random Forest (diabetes)')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------------------------------------
