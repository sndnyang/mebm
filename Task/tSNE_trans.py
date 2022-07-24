import matplotlib.pyplot as plt
from sklearn import manifold


def tsne_trans(all_features, epoch, args):
    vis_method = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = vis_method.fit_transform(all_features)
    colors = ['navy', 'turquoise', 'darkorange']
    plt.scatter(Y[:100, 0], Y[:100, 1], c=colors[0], cmap=plt.cm.Spectral)
    plt.scatter(Y[100:200, 0], Y[100:200, 1], c=colors[1], cmap=plt.cm.Spectral)
    if len(all_features) > 200:
        plt.scatter(Y[200:, 0], Y[200:, 1], c=colors[2], cmap=plt.cm.Spectral)
    # plt.set_title("%s (%.2g sec)" % (label, t1 - t0))
    plt.axis('tight')
    plt.savefig(args.save_dir + f"/tSNE_{epoch}.png")
    plt.close()
