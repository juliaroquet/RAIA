import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from util import plot_decision_regions
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns


font_path = r"C:\Users\Júlia\Documents\4T ANY TELECOS-TELEMATICA\RAIA\Retro_Dolly.ttf"
prop = fm.FontProperties(fname=font_path)


# Carreguem el dataset Telecom_churn
file_path = r"C:\Users\Júlia\Documents\4T ANY TELECOS-TELEMATICA\RAIA\datasets\Telecom_Churn.csv"
df = pd.read_csv(file_path)


# Seleccionem les dos variables més influenciables
X = df[['Total day charge','Customer service calls']]
y = df ['Churn'].astype(int)  #converteix True i False a 1 i 0

# Separem en train i test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1, stratify=y )
print('Class labels:', np.unique(y))
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train)) 
print('Labels counts in y_test:', np.bincount(y_test))



# KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# metrics https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

# model train metric='euclidean', n_neighbors=1
knn_model = KNeighborsClassifier(metric='euclidean', n_neighbors=5)
knn_model.fit(X_train, y_train)

# test prediction
y_pred = knn_model.predict(X_test)

misclassified = (y_test != y_pred).sum()
accuracy = knn_model.score(X_test, y_test)

print(f'Misclassified samples: {misclassified}')
print(f'Aciertos: {accuracy*100:.2f}%')



# ____________GRÀFIQUES_________________________
# DISTRIBUCIÓ DE CLASSES

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ['#ffb6c1']  # rosa clar i fosc
axes[0].bar(['No Churn','Churn'], np.bincount(y), color=colors)
axes[0].set_title("Distribucion total",color='#d63384',fontproperties=prop)
axes[1].bar(['No Churn','Churn'], np.bincount(y_train), color=colors)
axes[1].set_title("Train set",color='#d63384',fontproperties=prop)
axes[2].bar(['No Churn','Churn'], np.bincount(y_test), color=colors)
axes[2].set_title("Test set",color='#d63384',fontproperties=prop)
plt.tight_layout()
plt.show()

# MATRIU DE CONFUSIÓ
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap=ListedColormap(["#ffb6c1", "#ff69b4"]))
plt.title("Matriz de Confusion", color='#d63384',fontproperties=prop)
plt.show()

# Aciertos AND ERRORS
plt.figure(figsize=(5,3))
plt.text(0.5, 0.6, f"Aciertos: {accuracy*100:.2f}%", fontsize=20, ha='center', color='#d63384', fontproperties=prop)
plt.text(0.5, 0.4, f"Errores: {misclassified}", fontsize=20, ha='center', color='#ff69b4', fontproperties=prop)
plt.axis('off')
plt.show()

#GRAFICA DE CORRELACION
corr = df.corr(numeric_only=True)

# Dibujar mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="RdPu", fmt=".2f")
plt.title("Mapa de correlacion",color='#ff69b4', fontproperties=prop)
plt.show()

# Frontera de decisió, el algoritme mira els veïns més propers
# Segons a quina classi pertanyen la majoria dels veïns es classificaran en el nou punt
# La zona blava es on està la classe 1 Churm
# La zona taronja es on està la classe 0 no Churm
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
train_len = X_train.shape[0]
combined_len = X_combined.shape[0]

plt.figure(figsize=(6, 6), dpi=300)
plot_decision_regions(X=X_combined, y=y_combined, classifier=knn_model, test_idx=range(train_len, combined_len))
plt.xlabel('Total day charge',color='#ff69b4', fontproperties=prop) #trucades diaries
plt.ylabel('Customer service calls',color='#ff69b4', fontproperties=prop) #trucades al servei al client
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# el que veiem al gràfic es:
# quadrats vermells (0), clients que no han abandonat
# creus blaves (1) clients que sí que han abandonat

