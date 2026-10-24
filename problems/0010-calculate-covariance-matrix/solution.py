import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	vectors = np.array(vectors, dtype = np.float32)
	covariance = np.zeros((vectors.shape[-2], vectors.shape[-2]))
	num_samples = vectors.shape[-1]
	num_samples = num_samples-1 if num_samples ==1  else num_samples
	nf = covariance.shape[-1]
	for i in range(nf):
		for j in range(nf):
			sums = 0.0
			for k in range(num_samples):
				# print(vectors[i,:])
				# print(vectors[j,:])
				mean_i = np.mean(vectors[i,:])
				mean_j = np.mean(vectors[j,:])
				sums+= (vectors[i][k]-mean_i)*(vectors[j][k]-mean_j)
			# print(sums),
			
			covariance[i][j] = sums / (num_samples-1)


	return covariance