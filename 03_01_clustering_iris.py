import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Cargar el dataset de clientes
# <-- MODIFICADO: Cambiamos el nombre del archivo.
customers = pd.read_csv("customers_clustering.csv")

# Seleccionar las columnas 'Annual Income (k$)' y 'Spending Score (1-100)'
# <-- MODIFICADO: Usamos las columnas 3 y 4. Usamos .values para obtener un array de NumPy.
samples = customers.iloc[:, 3:5].values

# --- Método del Codo para encontrar el número óptimo de clusters ---
# K-Means requiere que especifiquemos el número de clusters. El método del codo
# es una técnica para encontrar un buen valor de 'k'.
inertia = []
K = range(1, 11) # Probaremos con k desde 1 hasta 10
for k in K:
    # Creamos y entrenamos el modelo
    kmeanModel = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeanModel.fit(samples)
    inertia.append(kmeanModel.inertia_)

# Graficamos el resultado del método del codo
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia')
plt.title('Método del Codo para Encontrar k Óptimo')
plt.show()
# NOTA: En el gráfico, el "codo" (el punto donde la tasa de descenso se ralentiza)
#       sugiere el mejor k. Para este dataset, suele ser k=5.

# --- Aplicamos K-Means con el k óptimo (k=5) ---
# <-- MODIFICADO: Usamos n_clusters=5 basado en el método del codo.
model = KMeans(n_clusters=5, n_init='auto', random_state=42)
model.fit(samples)

# Imprimimos las coordenadas de los centroides y las etiquetas de los clusters
print("Coordenadas de los centroides de los clusters:")
print(model.cluster_centers_)
# print("\nEtiquetas de cluster para cada dato:") # Descomentar si quieres ver todas las etiquetas
# print(model.labels_)

# --- Visualizamos los clusters y los centroides ---
# <-- MODIFICADO: Actualizamos las variables y etiquetas para el nuevo dataset.
plt.figure(figsize=(10, 7))
plt.scatter(samples[:,0], samples[:,1], c=model.labels_, cmap='viridis', s=50, alpha=0.7)
# Graficamos los centroides de cada cluster en negro
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color='black', marker='X', s=200, label='Centroides')

plt.title('Segmentación de Clientes con K-Means')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación de Gasto (1-100)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Predecir el cluster para nuevos clientes ---
# <-- MODIFICADO: Creamos ejemplos de nuevos clientes con [Ingreso, Gasto].
print("\n--- Predicción para nuevos clientes ---")
new_samples = np.array([
    [30, 80],  # Cliente con bajo ingreso y alto gasto
    [60, 50],  # Cliente promedio
    [100, 20]  # Cliente con alto ingreso y bajo gasto
])

new_labels = model.predict(new_samples)

for i, sample in enumerate(new_samples):
    print(f"El cliente con datos {sample} pertenece al cluster: {new_labels[i]}")