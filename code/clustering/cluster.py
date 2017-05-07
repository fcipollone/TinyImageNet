from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

X = np.load('/mnt/final_encoded_files/encoded0.npz')['data']
for i in xrange(1,16):
	print i
	filen = '/mnt/final_encoded_files/encoded' + str(i) + '.npz'
	X = np.concatenate((X,np.load(filen)['data']),axis=0)

#kmeans = KMeans(n_clusters=3, random_state=0).fit(X)


pca = PCA(n_components = 3).fit(X)

t = pca.transform(X)

np.savez('pcad',data=t)

#plt.plot(t[:,0],t[:,1])
#plt.show()
#np.savez('data/kmeans_labels',data=kmeans.labels_)
#np.savez('data/kmeans_clusters',data=kmeans.cluster_centers_)
