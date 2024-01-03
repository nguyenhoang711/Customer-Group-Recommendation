from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def k_analysis(data):
    wcss=[]
    for c in range(2, 11):
        kmeans=KMeans(n_clusters=c, random_state=0).fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(2, 11),wcss, marker='o',linestyle = '-.',color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('K-means Clustering')
    plt.show()