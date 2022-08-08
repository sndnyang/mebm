import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold


def tsne_trans(all_features, epoch, args, proj=False):
    np.save(args.save_dir + f'/features/features_{proj}_{epoch}.npy', all_features)
    if not proj:
        vis_method = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = vis_method.fit_transform(all_features)
    else:
        Y = all_features
    colors = ['navy', 'turquoise', 'darkorange']
    plt.scatter(Y[:200, 0], Y[:200, 1], c=colors[0], cmap=plt.cm.Spectral, label='$x^+$')
    plt.scatter(Y[200:400, 0], Y[200:400, 1], c=colors[1], cmap=plt.cm.Spectral, label='$x^0$')
    if len(all_features) > 400:
        plt.scatter(Y[400:, 0], Y[400:, 1], c=colors[2], cmap=plt.cm.Spectral, label='$x^-$')
    # plt.set_title("%s (%.2g sec)" % (label, t1 - t0))
    plt.axis('tight')
    plt.savefig(args.save_dir + f"/features/tSNE_{proj}_{epoch}.png")
    plt.close()


def tsne_trace(all_feats, sampling_feats, steps, epoch, args, proj=False):
    np.save(args.save_dir + f'/tracing/features_{proj}_{epoch}.npy', sampling_feats)
    all_features = np.concatenate([all_feats, sampling_feats])
    if not proj:
        np.savetxt('{}/tracing/{}_step.txt'.format(args.save_dir, epoch), steps, fmt='%d', delimiter=",")
        vis_method = manifold.TSNE(n_components=2, init='pca', random_state=0)
        Y = vis_method.fit_transform(all_features)
    else:
        Y = all_features
    colors = ['navy', 'turquoise', 'darkorange']

    fig, ax = plt.subplots()
    ax.scatter(Y[:200, 0], Y[:200, 1], c=colors[0], cmap=plt.cm.Spectral, label='$x^+$')
    ax.scatter(Y[200:400, 0], Y[200:400, 1], c=colors[1], cmap=plt.cm.Spectral, label='$x^0$')
    ax.scatter(Y[400:464, 0], Y[400:464, 1], c=colors[2], cmap=plt.cm.Spectral, label='$x^-$')

    plt.axis('tight')
    plt.legend()
    plt.savefig(args.save_dir + f"/tracing/tSNE_{proj}_{epoch}.png")

    # only show a new array
    index = np.argmin(steps)

    feat_by_step = np.concatenate([Y[464:], Y[400:464]]).reshape(args.batch_size, args.n_steps + 1, 2)
    scratch_sampling = feat_by_step[index:index+1][0]
    x = scratch_sampling[:, 0]
    y = scratch_sampling[:, 1]
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u / 2
    pos_y = y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)
    ax.plot(x, y, c='red', marker="o", linewidth=0.5)
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid", color='red', label=f'{np.min(steps)}')
    ax.scatter(x[:1], y[:1], c='black')
    start_points = [(x[0], y[0])]

    index_max = np.argmax(steps)
    scratch_sampling = feat_by_step[index_max:index_max+1][0]
    x = scratch_sampling[:, 0]
    y = scratch_sampling[:, 1]
    start_points += [(x[0], y[0])]
    u_max = np.diff(x)
    v_max = np.diff(y)
    pos_x_max = x[:-1] + u_max / 2
    pos_y_max = y[:-1] + v_max / 2
    norm_max = np.sqrt(u_max ** 2 + v_max ** 2)
    ax.plot(x, y, c='green', marker="o", linewidth=0.5)

    ax.quiver(pos_x_max, pos_y_max, u_max / norm_max, v_max / norm_max, angles="xy", zorder=5, pivot="mid", color='green', label=f'{np.max(steps)}')
    ax.scatter(x[:1], y[:1], c='black')
    plt.legend()
    plt.title(f'Start points {str(start_points)}')
    plt.savefig(args.save_dir + f"/tracing/tSNE_{proj}_arrow_{epoch}.png")
    plt.close()
