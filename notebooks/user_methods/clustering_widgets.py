# coding: utf-8
import warnings

import matplotlib.pyplot as plt
import numpy as np


def plot_kmeans_interactive(min_clusters=1, max_clusters=6):
    from ipywidgets import interact
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.datasets.samples_generator import make_blobs

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        X, y = make_blobs(n_samples=300, centers=4,
                          random_state=0, cluster_std=0.60)

        def _kmeans_step(frame=0, n_clusters=4):
            rng = np.random.RandomState(2)
            labels = np.zeros(X.shape[0])
            centers = rng.randn(n_clusters, 2)

            nsteps = frame // 3

            for i in range(nsteps + 1):
                old_centers = centers
                if i < nsteps or frame % 3 > 0:
                    dist = euclidean_distances(X, centers)
                    labels = dist.argmin(1)

                if i < nsteps or frame % 3 > 1:
                    centers = np.array([X[labels == j].mean(0)
                                        for j in range(n_clusters)])
                    nans = np.isnan(centers)
                    centers[nans] = old_centers[nans]

            # plot the data and cluster centers
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='rainbow',
                        vmin=0, vmax=n_clusters - 1);
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c=np.arange(n_clusters),
                        s=200, cmap='rainbow')
            plt.scatter(old_centers[:, 0], old_centers[:, 1], marker='o',
                        c='black', s=50)

            # plot new centers if third frame
            if frame % 3 == 2:
                for i in range(n_clusters):
                    plt.annotate('', centers[i], old_centers[i],
                                 arrowprops=dict(arrowstyle='->', linewidth=1))
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c=np.arange(n_clusters),
                            s=200, cmap='rainbow')
                plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c='black', s=50)

            plt.xlim(-4, 4)
            plt.ylim(-2, 10)

            if frame % 3 == 1:
                plt.text(3.8, 9.5, "1. Reassign points to nearest centroid",
                         ha='right', va='top', size=14)
            elif frame % 3 == 2:
                plt.text(3.8, 9.5, "2. Update centroids to cluster means",
                         ha='right', va='top', size=14)

    return interact(_kmeans_step, frame=[0, 50],
                    n_clusters=[min_clusters, max_clusters])


def plot_image_components(x, coefficients=None, mean=0, components=None,
                          imshape=(8, 8), n_components=6, fontsize=12):
    if coefficients is None:
        coefficients = x

    if components is None:
        components = np.eye(len(coefficients), len(x))

    mean = np.zeros_like(x) + mean

    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 5 + n_components, hspace=0.3)

    def show(i, j, x, title=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest')
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(slice(2), slice(2), x, "True")

    approx = mean.copy()
    show(0, 2, np.zeros_like(x) + mean, r'$\mu$')
    show(1, 2, approx, r'$1 \cdot \mu$')

    for i in range(0, n_components):
        approx = approx + coefficients[i] * components[i]
        show(0, i + 3, components[i], r'$c_{0}$'.format(i + 1))
        show(1, i + 3, approx,
             r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1))
        plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
                       transform=plt.gca().transAxes, fontsize=fontsize)

    show(slice(2), slice(-2, None), approx, "Approx")


def plot_pca_interactive(data, n_components=6):
    from sklearn.decomposition import PCA
    from IPython.html.widgets import interact

    pca = PCA(n_components=n_components)
    Xproj = pca.fit_transform(data)

    def show_decomp(i=0):
        plot_image_components(data[i], Xproj[i],
                              pca.mean_, pca.components_)

    interact(show_decomp, i=(0, data.shape[0] - 1));


def plot_dendrogram_interactive(linkage_method='complete', color_thresh=None, df=None):
    '''
    Builds a linkage for a dendrogram

    :param str linkage_method: The name of the linkage method ('single', 'complete', 'average', 'centroid', 'weighted', 'median', 'ward')
    :param double color_thresh: A colour threshold for the cluster nodes
    :param dataframe df: A dataframe with index
    :return: the dendrogram object
    '''

    import scipy
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import cluster
    from ipywidgets import interact, fixed, widgets

    # Data set
    if df is None:
        url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
        df = pd.read_csv(url)
        df = df.set_index('model')
        del df.index.name

    # Calculate the distance between each sample
    Z = cluster.hierarchy.linkage(df, method=linkage_method)

    if color_thresh is None:
        P = cluster.hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=df.index, no_plot=True,
                                         distance_sort=True)
    else:
        P = cluster.hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=df.index, no_plot=True,
                                         distance_sort=True, color_threshold=color_thresh)

    def _plot_dendrogram(pos=None, P=None):
        '''
        Builds a dendrogram interactively

        :param tuple pos: The first and the last positions of the step slider, e.g. (1, 20)
        :param linkage P: A dendrogram object
        :return: the widget
        '''
        import numpy as np
        
        plt.figure(figsize=(15, 10))

        #     plot_tree(P, list(range(20)))

        icoord = scipy.array(P['icoord'])
        dcoord = scipy.array(P['dcoord'])
        color_list = scipy.array(P['color_list'])
        xmin, xmax = icoord.min(), icoord.max()
        ymin, ymax = dcoord.min(), dcoord.max()

        if pos is not None:
            pos = np.argsort(dcoord[:, 1])[:pos]
            icoord = icoord[pos]
            dcoord = dcoord[pos]
            color_list = color_list[pos]

        for xs, ys, color in zip(icoord, dcoord, color_list):
            plt.plot(xs, ys, color)

        plt.xlim(xmin - 10, xmax + 0.1 * abs(xmax))
        plt.ylim(ymin, ymax + 0.1 * abs(ymax))
        icoord = scipy.array(P['icoord'])[:, [0, 3]].reshape(1, -1)
        dcoord = scipy.array(P['dcoord'])[:, [0, 3]].reshape(1, -1)

        x_ticks = icoord[dcoord == 0]
        x_ticks.sort()
        plt.xticks(x_ticks, P['ivl'], rotation='vertical')
        plt.show()

    return interact(_plot_dendrogram, pos=widgets.IntSlider(min=1, max=31, step=1, value=1), P=fixed(P))
