
def plot_dendrogram_interactive(linkage_method = 'complete', color_thresh = None, df = None):
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
    if df == None:
        url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
        df = pd.read_csv(url)
        df = df.set_index('model')
        del df.index.name
    
    # Calculate the distance between each sample
    Z = cluster.hierarchy.linkage(df, method = linkage_method)
    
    if color_thresh == None:
        P = cluster.hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=df.index, no_plot=True,
                                    distance_sort = True)
    else:
        P = cluster.hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=10, labels=df.index, no_plot=True,
                                    distance_sort = True, color_threshold = color_thresh)
    
    def _plot_dendrogram(pos = None, P = None):
        '''
        Builds a dendrogram interactively

        :param tuple pos: The first and the last positions of the step slider, e.g. (1, 20)
        :param linkage P: A dendrogram object
        :return: the widget
        '''

        pos = range(0, pos)

        plt.figure(figsize=(15, 10))

    #     plot_tree(P, list(range(20)))

        icoord = scipy.array( P['icoord'] )
        dcoord = scipy.array( P['dcoord'] )
        color_list = scipy.array( P['color_list'] )
        xmin, xmax = icoord.min(), icoord.max()
        ymin, ymax = dcoord.min(), dcoord.max()

        if pos:
            icoord = icoord[pos]
            dcoord = dcoord[pos]
            color_list = color_list[pos]

        for xs, ys, color in zip(icoord, dcoord, color_list):
            plt.plot(xs, ys,  color)


        plt.xlim( xmin-10, xmax + 0.1*abs(xmax) )
        plt.ylim( ymin, ymax + 0.1*abs(ymax) )
        icoord = scipy.array( P['icoord'] )[:, [0, 3]].reshape(1, -1)
        dcoord = scipy.array( P['dcoord'] )[:, [0, 3]].reshape(1, -1)

        x_ticks = icoord[dcoord == 0]
        x_ticks.sort()
        plt.xticks(x_ticks, P['ivl'], rotation='vertical')    
        plt.show()
        
    return interact(_plot_dendrogram, pos=widgets.IntSlider(min=1,max=31,step=1,value=1), P=fixed(P))
    

