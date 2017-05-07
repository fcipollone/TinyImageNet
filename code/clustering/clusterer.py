import numpy as np
from sklearn.metrics import mean_squared_error


def cluster(Hr_vals):
	clusters = np.load('data/kmeans_clusters.npz')['data']
	res = []
	print len(Hr_vals)
	for val in Hr_vals:
		val = np.reshape(val,(1000))
		min_cost = None
		min_cluster = None
		for i in xrange(0,len(clusters)):
			d = mean_squared_error(val,clusters[i])
			if min_cost == None:
				min_cost = d
				min_cluster = i
			else:
				if d < min_cost:
					min_cost = d
					min_cluster = i
		res.append(min_cluster)


	return res

