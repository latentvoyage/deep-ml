import numpy as np
import math

class Stump:
	def __init__(self):
		self.p = 1
		
		self.j = -1
		self.t = -1

	def fit(self, X, y,w):
		_,n_features = np.shape(X)
		t_star = -1
		j_star = -1
		min_error = 1e9

		for j in range(n_features):
			t_samples = np.unique(X[:,j])
			# for each t, identify if the current value is the minimum error
			for t in t_samples:
				preds = np.where(X[:,j]>=t, 1,-1)
				h_x = self.p*preds
				e = w.T @ (h_x!=y)
				p = 1
				if e > 0.5:
					e = 1 - e
					p = -1

				if e < min_error:
					min_error = e
					t_star = t
					j_star = j
					p_star = p

		self.j = j_star
		self.t = t_star
		self.p = p_star
		return min_error, t_star, j_star
	
	def predict(self,X):
		preds = np.where(X[:,self.j]>=self.t, 1,-1)
		h_x = self.p*preds
		return h_x

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples))

	clfs = []

	# Your code here
	# for each timestep
	# calculate number of errors of the current classifier
	# compute alpha t 
	# compute weights of each input point and then fit the next classifier based on the weighted zero-one loss
	for _ in range(n_clf):
		# fit the classifier
		stump = Stump()
		e, t, j = stump.fit(X,y,w)
		alpha = 0.5 * np.log((1-e)/(e+(10**(-10))))
		record = {}
		record['polarity'], record['threshold'], record['feature_index'], record['alpha'] = stump.p, stump.t, stump.j, alpha
		clfs.append(record)
		h_x = stump.predict(X)
		w = w * np.exp(-alpha * y * h_x)
		w /= w.sum()
		
		

	return clfs
    