import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	points = np.array(points, dtype=float)
	centroids = np.array(initial_centroids, dtype=float)

	def measure(point1, point2):
		return np.linalg.norm(point1 - point2, ord=2)

	for _ in range(max_iterations):

		# Assignment step
		labels = np.array([
			np.argmin([measure(c, point) for c in centroids])
			for point in points
		])

		# Update step
		for i in range(k):
			cluster_points = points[labels == i]

			if len(cluster_points) > 0:
				centroids[i] = np.mean(cluster_points, axis=0)

	return [tuple(c) for c in centroids]