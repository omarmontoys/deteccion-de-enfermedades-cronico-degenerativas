from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from requests import head
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer, SimpleImputer
#-----------------------------------------------------------------------------------

# Simbologia 

# valor_COL_HDL   --------------------------------  Colesteron	
# valor_GLU_SUERO --------------------------------  Glucosa
# valor_VIT_B12	  --------------------------------  Vitamina B12
# valor_VIT_D	  --------------------------------  Vitamina D
# an09            --------------------------------  Medicion de la cintura
# an30            --------------------------------  Tension Arterial
# an03            --------------------------------  Medicion de Peso y ropa puesta.
# h0302           --------------------------------  Sexo
# h0303           --------------------------------  Edad
# fa0400          --------------------------------  Horas promedio de sueño
# fa0407h         --------------------------------  Tiempo sentado a la semana en horas


#-----------------------------------------------------------------------------------
#Modificamos las reglas de como imprimir los datos (Imprimir .. o  Imprimir sin limitacion)

#pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')
#-----------------------------------------------------------------------------------
#Leemos los dataset 

# Leer el archivo CSV directamente con el delimitador ';'
df1 = pd.read_csv('bioquimicas.csv', delimiter=';')
# Mostrar el DataFrame
print(df1)
# Leer el archivo CSV directamente con el delimitador ';'
df2 = pd.read_csv('ensaantro2022_entrega_w.csv', delimiter=';')
# Mostrar el DataFrame
print(df2)
# Leer el archivo CSV directamente con el delimitador ';'
df3 = pd.read_csv('ensafisica2022_adultos_entrega_w.csv', delimiter=';')
# Mostrar el DataFrame
print(df3)

#-----------------------------------------------------------------------------------
#Categorizamos los espacios en blanco para que sean reconocidos como NaN

df1.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df2.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df3.replace(r'^\s*$', np.nan, regex=True, inplace=True)

#-----------------------------------------------------------------------------------

# Imprimimos los dataframe con los espacion en blanco convertidos en NaN

print(df1)
print(df2)
print(df3)

#-----------------------------------------------------------------------------------

#Calculamos el numero de nulos por columna de cada dataset
print("\nTablas comprobando el numero de nulos por dataset :")

#Mostramos una tabla con la cantidad de nulos por columna
cantidadnull = df1.isnull().sum()
print("\nCantidad de nulos por columa dataset1:")
print(cantidadnull)
#Mostramos una tabla con la cantidad de nulos por columna
cantidadnull = df2.isnull().sum()
print("\nCantidad de nulos por columa dataset2:")
print(cantidadnull)
#Mostramos una tabla con la cantidad de nulos por columna
cantidadnull = df3.isnull().sum()
print("\nCantidad de nulos por columa dataset3:")
print(cantidadnull)

#-----------------------------------------------------------------------------------
#Eliminamos las filas duplicadas de cada dataframe.

# Eliminar filas duplicadas basándose en todas las columnas
df_sin_duplicados = df1.drop_duplicates()
# Mostrar el DataFrame 1 sin duplicados
print(df_sin_duplicados)
# Eliminar filas duplicadas basándose en todas las columnas
df_sin_duplicados = df2.drop_duplicates()
# Mostrar el DataFrame 2 sin duplicados
print(df_sin_duplicados)
# Eliminar filas duplicadas basándose en todas las columnas
df_sin_duplicados = df3.drop_duplicates()
# Mostrar el DataFrame 3 sin duplicados
print(df_sin_duplicados)

#-----------------------------------------------------------------------------------
print("\nNuevo dataset usando las columnas mas importantes:")
# Selecciona las columnas deseadas de cada DataFrame
df1_selected = df1[['valor_COL_HDL', 'valor_GLU_SUERO', 'valor_VIT_B12', 'valor_VIT_D']]
df2_selected = df2[['an09', 'an30','an03']]
df3_selected = df3[['h0302', 'h0303', 'fa0400', 'fa0407h']]

# Concatena los DataFrames a lo largo de las filas (eje=0)
dfFinal = pd.concat([df1_selected, df2_selected, df3_selected], ignore_index=True)

# Muestra el nuevo DataFrame resultante
print(dfFinal)

#-----------------------------------------------------------------------------------

# Reemplazar comas por puntos en columnas específicas
columns_to_replace_commas = ['valor_COL_HDL', 'valor_GLU_SUERO', 'valor_VIT_B12', 'valor_VIT_D','an09','an03', 'an30','h0302', 'h0303', 'fa0400', 'fa0407h']  # Reemplaza con los nombres reales de las columnas
dfFinal[columns_to_replace_commas] = dfFinal[columns_to_replace_commas].replace({',': '.'}, regex=True)
# Convertir las columnas relevantes a números de punto flotante
dfFinal[columns_to_replace_commas] = dfFinal[columns_to_replace_commas].astype(float)

#-----------------------------------------------------------------------------------
# Imputar los valores faltantes utilizando la media de cada columna
imputer = SimpleImputer(strategy='mean')
dfFinal_imputed = pd.DataFrame(imputer.fit_transform(dfFinal), columns=dfFinal.columns)

# Verificar el nuevo DataFrame con valores imputados
print("\nNuevo DataFrame con valores imputados:")
print(dfFinal_imputed)
#dfFinal_imputed.to_csv('dfFinal_imputed.csv', index=False)
#-----------------------------------------------------------------------------------
#Calculamos la matriz de correlacion
dfCorre = dfFinal_imputed.corr(method="pearson")
# Imprimir la matriz de correlación
print("\nMatriz de correlación:")
print(dfCorre)

#-----------------------------------------------------------------------------------
# Seleccionar características
feature_columns = ['valor_COL_HDL', 'valor_GLU_SUERO', 'valor_VIT_B12', 'valor_VIT_D', 'an09', 'an03', 'an30', 'h0302', 'h0303', 'fa0400', 'fa0407h']
df_features = dfFinal_imputed[feature_columns]

# Estandarizar características
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

# Aplicar KMeans
kmeans_model = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans_model.fit_predict(df_scaled)

# Visualización con KMeans (ejemplo con las dos primeras características)
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=kmeans_labels, cmap='viridis', 
             marker='o', edgecolor='k', s=50)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#-----------------------------------------------------------------------------------



