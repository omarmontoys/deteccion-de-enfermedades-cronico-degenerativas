from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_curve 
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

# Leer los datasets
df1 = pd.read_csv('bioquimicas.csv', delimiter=';',skipinitialspace=True)
df2 = pd.read_csv('ensaantro2022_entrega_w.csv', delimiter=';',skipinitialspace=True)
df3 = pd.read_csv('ensafisica2022_adultos_entrega_w.csv', delimiter=';',skipinitialspace=True)
df4 = pd.read_csv('ensadul2022_entrega_w.csv', delimiter=';',skipinitialspace=True)
#df5 = pd.read_csv('ensafrec_ad2022_entrega_w.csv', delimiter=';')

#Obtener folios únicos para los conjuntos de datos por separado
folios_unicos_df1 = len(df1['FOLIO_INT'].unique())
folios_unicos_df2 = len(df2['FOLIO_INT'].unique())
folios_unicos_df3 = len(df3['FOLIO_INT'].unique())
folios_unicos_df4 = len(df4['FOLIO_INT'].unique())
#folios_unicos_df5 = len(df5['FOLIO_INT'].unique())

#Mostrar la cantidad de folios únicos por conjunto de datos
print("Cantidad de folios únicos en el primer Dataset:", folios_unicos_df1)
print("Cantidad de folios únicos en el segundo Dataset:", folios_unicos_df2)
print("Cantidad de folios únicos en el tercer Dataset:", folios_unicos_df3)
print("Cantidad de folios únicos en el cuarta Dataset:", folios_unicos_df4)
#print("Cantidad de folios únicos en el quinta Dataset:", folios_unicos_df5)

# Combinar los cinco conjuntos de datos en base a la columna FOLIO_INT
merged_df = pd.merge(df1, df2, on='FOLIO_INT', how='inner', suffixes=('_df1', '_df2'))
merged_df = pd.merge(merged_df, df3, on='FOLIO_INT', how='inner', suffixes=('', '_df3'))
merged_df = pd.merge(merged_df, df4, on='FOLIO_INT', how='inner', suffixes=('', '_df4'))
#merged_df = pd.merge(merged_df, df4, on='FOLIO_INT', how='inner', suffixes=('', '_df5'))
print(merged_df)

# Selecciona las columnas deseadas de cada DataFrame
df1_selected = merged_df[['a0401', 'a0301','h0302', 'h0303', 'fa0400', 'fa0407h','an09', 'an30','an03']]
#df5_selected = merged_df[['h0302', 'h0303', 'fa0400', 'fa0407h']]


# Muestra el nuevo DataFrame resultante
print(df1_selected)

cantidadnull = df1_selected.isnull().sum()
print("\n Se encontraron valores nulos en el DataFrame. \n", cantidadnull)

dfFinalNull = df1_selected.dropna()

print("DataFrame después de dropna:")
print(dfFinalNull)
dfFinalNull.to_csv('dfFloat.csv', index=False)


# Eliminar filas duplicadas basándose en todas las columnas
#df_sin_duplicados = dfFinalNull.drop_duplicates()
# Mostrar el DataFrame 3 sin duplicados
#print(df_sin_duplicados)
# Reemplazar comas por puntos en las columnas relevantes
columns_to_replace_commas = [ 'a0401', 'a0301', 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03']
dfFinalNull[columns_to_replace_commas] = dfFinalNull[columns_to_replace_commas].replace({',': '.'}, regex=True)

# Convertir las columnas relevantes a números de punto flotante
dfFloat = dfFinalNull

#-----------------------------------------------------------------------------------
# Calculamos la matriz de correlacion
dfCorre = dfFloat.corr(method="pearson")
# Imprimir la matriz de correlación
print("\nMatriz de correlación:")
print(dfCorre)

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
# Almacena la suma de los cuadrados de las distancias para cada número de clusters
sum_of_squared_distances = []
# Prueba con un número de clusters en un rango específico
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sum_of_squared_distances.append(kmeans.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Número de clusters')
plt.ylabel('Suma de cuadrados de las distancias')
plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', 
            marker='o', edgecolor='k', s=50)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = dfFloat.drop(columns=[ 'h0302', 'h0303', 'fa0400', 'fa0407h', 'an09', 'an30', 'an03'])  # Ajusta las columnas según tu necesidad
y = dfFloat['a0301']


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar y entrenar el modelo Adaline (SGDRegressor)
adaline_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42)
adaline_model.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = adaline_model.predict(X_test_scaled)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Estandarizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de regresión logística con Gradiente Descendente Estocástico
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Evaluar el modelo
accuracy = logistic_model.score(X_test, y_test)
y_pred = logistic_model.predict(X_test)
# Calcular la precisión
precision = precision_score(y_test, y_pred)
# Calcular la sensibilidad (recall)
sensitivity = recall_score(y_test, y_pred)
y_prob = logistic_model.predict_proba(X_test)[:, 1]  # Probabilidad de clase positiva
roc_auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy del modelo: {accuracy}')
print(f'Precision del modelo: {precision}')
print(f'Sensibilidad del modelo: {sensitivity}')
print(f'AUC del ROC modelo: {roc_auc}')
# Visualizar la curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Inicializar y entrenar el modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=3)  # Puedes ajustar el valor de n_neighbors según sea necesario
knn_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = knn_model.predict(X_test)

# Calcular métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'ROC AUC: {roc_auc}')
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')
plt.legend(loc='lower right')
plt.show()