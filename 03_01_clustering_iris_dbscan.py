import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# Cargar el nuevo dataset de clientes
# <-- MODIFICADO: Cambiamos el nombre del archivo.
customers = pd.read_csv("customers_clustering.csv")

# Seleccionar las columnas 'Annual Income (k$)' y 'Spending Score (1-100)'
# <-- MODIFICADO: Seleccionamos las columnas 3 y 4 del nuevo dataset.
selected_data = customers.iloc[:, 3:5]

# Estandarizamos los datos (esto sigue siendo una buena práctica)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(selected_data)

# Aplicamos DBSCAN. Los parámetros eps y min_samples pueden necesitar ajuste
# para este nuevo dataset, pero podemos empezar con los mismos.
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Visualizamos los clusters
# <-- MODIFICADO: Apuntamos a las columnas correctas del dataframe 'customers'.
x = customers.iloc[:, 3] # Columna 'Annual Income (k$)'
y = customers.iloc[:, 4] # Columna 'Spending Score (1-100)'
plt.scatter(x, y, c=clusters, cmap='rainbow')

# <-- MODIFICADO: Actualizamos las etiquetas de los ejes para que coincidan con los datos.
plt.title('Segmentación de Clientes con DBSCAN')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Información sobre los clusters generados
# El cluster -1 representa el "ruido" (puntos que no pertenecen a ningún grupo).
print(f'Número de clusters encontrados (incluyendo ruido): {len(np.unique(clusters))}')
print(f'Etiquetas de los clusters: {np.unique(clusters)}')