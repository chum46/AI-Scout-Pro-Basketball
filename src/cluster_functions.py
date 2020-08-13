import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import cluster, metrics, datasets

from scipy.spatial.distance import cdist
import matplotlib.patches as mpatches

def kmeans(reduced_data, n_clusters):
    """
    performs kmeans clustering and returns labels, centroids, inertia, and silhouette score
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans = kmeans.fit(reduced_data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    sil_score = metrics.silhouette_score(reduced_data, kmeans.labels_, metric='euclidean')

    data_dictionary = {
        "labels": labels,
        "centroids": centroids,
        "inertia" : inertia,
        "silhouette_score": sil_score
    }

    return data_dictionary

def find_best_cluster(data,a,b):
    """
    plots and finds the best silhouette score for range(a,b)
    """
    scores = []
    distortions = []
    centroids = []
    for i in range(a,b):
        i_clusters = kmeans(data, i)
        sil_score_i = i_clusters['silhouette_score']
        scores.append(sil_score_i)
        distortions.append(i_clusters['inertia'])
        centroids.append(i_clusters['centroids'])
        print(i, sil_score_i)

    plt.plot(range(a,b), scores)
    plt.grid(which='minor')
    plt.xticks(np.arange(2,20,step=1))
    plt.title("Silhouette Score")
    plt.grid(True)
    plt.show
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Elbow curve')
    ax.set_xlabel('k')
    ax.set_xticks(np.arange(2,20,step=1))
    ax.plot(range(a,b), distortions)
    ax.grid(True)
    return

def plot_kmeans_cluster(reduced_data, k_clusters, plot_title):
    kmeans = KMeans(init='k-means++', n_clusters=k_clusters, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(15,10))
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.get_cmap("tab20"),
           aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=10)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    print(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title(plot_title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return

def plot_kmeans_circle (kmeans, X, n_clusters=6, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2, edgecolor='black')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1],
        marker='x', s=169, linewidths=5,
        color='w', zorder=10)
    plot_title="KMeans Clustering of NBA Players in 2008-2020"
    plt.title(plot_title)
    
    # plot the representation of the KMeans model
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))

def feature_importance(cluster_data, league_data, n):
    """
    takes reduced data,
    performs Principal Component Analysis,
    returns feature importance dataframe
    """
    X = cluster_data[['Raptor+/-', 'Raptor D','Raptor O','TS%','PIE%','ORtg','%Pos','3P%', '3PAr', 'FTAr', 'AST%', 'USG%', '2P%']]
    league = league_data[['Raptor+/-', 'Raptor D','Raptor O','TS%','PIE%','ORtg','%Pos','3P%', '3PAr', 'FTAr', 'AST%', 'USG%', '2P%']]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    PCA_reduced_df = pca.fit(scaled_data).transform(scaled_data)

    features = pd.DataFrame(zip(X.columns, pca.components_[0], np.mean(X), np.mean(league)),
        columns=['Feature', 'Importance', 'Cluster Average', 'League Average']).sort_values('Importance', ascending=False).head(20)

    return features

def bar_features (fi,cluster,color):
    fi['Difference']=(fi['Cluster Average']-fi['League Average'])/np.absolute(fi['League Average'])*100
    plt.rcParams['figure.figsize'] = [10, 6]
    plotdata = pd.DataFrame(
    {cluster: list(fi['Difference'])}, 
    index=fi['Feature'])
    plotdata.plot(kind='bar', color =color);