import random

# def generate_point(mean_x, mean_y, deviation_x, deviation_y):
#     return random.gauss(mean_x, deviation_x), random.gauss(mean_y, deviation_y)

def generate_point(means_devs: "list[tuple[float, float]]"):
    return tuple(random.gauss(mean, dev) for mean, dev in means_devs)

def str_point(point):
    return ','.join(map(str, point)) + "\n"

cluster_means = [100, 100, 100]
cluster_devs = [80, 30, 70]

point_devs = [40, 15, 50]

number_of_clusters = 5
points_per_cluster = 1_200_000 // 5


cluster_centers = [generate_point(list(zip(cluster_means, cluster_devs)))
                   for i in range(number_of_clusters)]

points = [generate_point(list(zip(center, point_devs)))
          for center in cluster_centers
          for i in range(points_per_cluster)]

with open(f"{number_of_clusters * points_per_cluster}.csv", "w") as out:
    out.write(','.join(f"c{i}" for i in range(len(cluster_means))) + "\n")
    out.writelines([str_point(point) for point in points])