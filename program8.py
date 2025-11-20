from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\acer\Downloads\P8Data.csv")
x = dataset.iloc[:,:-1]
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [label[c] for c in dataset.iloc[:,-1]]

plt.figure(figsize=(14,7))
colormap = np.array(['red','lime','black'])


plt.subplot(1,3,1)
plt.title("Real")
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y])

model = KMeans(n_clusters=3, random_state=3425).fit(x)
plt.subplot(1,3,2)
plt.title('KMeans')
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[model.labels_])
print('\nAccuracy score of KMeans:',metrics.accuracy_score(y,model.labels_))
print('\nConfusion matrix of KMeans:',metrics.confusion_matrix(y,model.labels_))

gmm = GaussianMixture(n_components=3, random_state=3425).fit(x)
y_gmm = gmm.predict(x)
plt.subplot(1,3,3)
plt.title('GMM Classification')
plt.scatter(x.Petal_Length, x.Petal_Width, c = colormap[y_gmm])
print('\nAccuracy score of KMeans:',metrics.accuracy_score(y,y_gmm))
print('\nConfusion matrix of KMeans:',metrics.confusion_matrix(y,y_gmm))

plt.tight_layout()
plt.show()
