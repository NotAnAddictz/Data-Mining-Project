#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


# In[87]:


df = pd.read_csv('./ClassicHit.csv')


# In[88]:


# Basic statistical details

df.describe()


# In[89]:


##['Year', 'Track', 'Artist','Genre', 'Danceability' , 'Energy', 'Liveness', 'Valence']
df = df.drop(['Year', 'Track', 'Artist', 'Genre','Time_Signature', 'Duration', 'Mode', 'Popularity'], axis= 1)


# In[ ]:





# In[90]:


df.isnull().sum()


# In[91]:


# Removing duplicate rows

print('Duplicate Rows Count : ', df.duplicated().sum())

df=df.drop_duplicates(keep="first")


# In[92]:


df.head()


# In[93]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)


# In[110]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fit PCA on scaled data
pca = PCA().fit(scaled_features)

# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.show()


# In[111]:


def pca(data, n):
    
    if type(n) == int:
        
        pca = PCA(n_components = n )
        pca.fit(data)
        df_pca = pca.transform(data)
        return df_pca
        
    else:
        return data


# In[115]:


##for simplicity of visualization
n_components = 3
pca = PCA(n_components=n_components)


# In[116]:


from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
 
neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(df)
distances, indices = neighbors_fit.kneighbors(df)


# In[117]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


# In[118]:


from sklearn.decomposition import PCA

# Applying PCA with the optimal number of components
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(scaled_features)
print(pca_result.shape)


# In[119]:


df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
sns.pairplot(df)
plt.show()


# In[130]:


from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps= 0.35, min_samples=10).fit(pca_result)
labels = clustering.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


# In[ ]:





# In[121]:


unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
        

    class_member_mask = labels == k

    xy = pca_result[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = pca_result[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()


# In[122]:


import matplotlib.pyplot as plt

# For a 2D plot (if n_components=2)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Results')
plt.show()

# For a 3D plot (if n_components=3)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D PCA Results')
plt.show()


# In[123]:


## Elbow method


# In[124]:



dataset=df.iloc[:,[1,2]].values
dataset

from sklearn.cluster import KMeans
WCSS=[]
for i in range(1,20):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(dataset)
    WCSS.append(kmeans.inertia_)
WCSS

plt.plot(range(1,20),WCSS)


# In[125]:


from sklearn.cluster import KMeans

# Determine the number of clusters
# (This number can be determined based on domain knowledge, heuristics, or methods like the Elbow Method)
n_clusters = 4

# Applying KMeans clustering
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_pca.fit(pca_result)

# The cluster labels for each data point
cluster_labels = kmeans_pca.labels_


# In[127]:


# 2D Visualization
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering on PCA Results')
plt.show()

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_labels)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('3D KMeans Clustering on PCA Results')
plt.show()


# In[ ]:





# In[ ]:




