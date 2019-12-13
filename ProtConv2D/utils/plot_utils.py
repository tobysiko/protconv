import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def plot_hist(
    y,
    scores,
    title="Binary class prediction",
    xlabel="probability",
    ylabel="#",
    size=(1.5, 1.5),
    nbins=10,
    poscolor="red",
    negcolor="blue",
    poslabel="true",
    neglabel="false",
    vline=0.5,
):
    fig = plt.figure(figsize=size, dpi=80)
    axes = fig.add_axes([0, 0, 1, 1])
    bins = np.linspace(0, 1, nbins + 1)
    axes.hist(
        [x[0] for x in zip(scores, y) if x[1] == 1],
        bins,
        alpha=0.5,
        color=poscolor,
        label=poslabel,
    )
    axes.hist(
        [x[0] for x in zip(scores, y) if x[1] == 0],
        bins,
        alpha=0.5,
        color=negcolor,
        label=neglabel,
    )
    axes.vlines(
        vline, 0, np.histogram(scores, bins)[0].max(), color="black", linestyles="--"
    )
    axes.set_ylim((0, np.histogram(scores, bins)[0].max()))
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    plt.legend(loc="best")
    return fig


def plot_ROC(observations, probabilities, title="", labels=True, size="auto"):
    """
    Creates ROC plot from observations (y_test) and probabilities (y_pred_proba)
    title -- title of the plot
    size -- tuple, size in inch, defaults to 'auto'
    labels -- toogle display of title and x and y labels and tick labels
    """
    if size is "auto":
        fig = plt.figure()
    else:
        fig = plt.figure(num=None, figsize=size, dpi=80)
    axes = fig.add_axes([0, 0, 1, 1])
    fpr, tpr, thresholds = roc_curve(observations, probabilities)
    axes.plot(fpr, tpr)
    axes.plot([0, 1], [0, 1], "k--")
    axes.set_aspect("equal")
    if labels:
        axes.set_title(title)
        axes.set_xlabel("False Positive Rate")
        axes.set_ylabel("True Positive Rate")
    else:
        axes.get_xaxis().set_ticks([])
        axes.get_yaxis().set_ticks([])
    return fig
def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    size=(3, 3),
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig = plt.figure(figsize=size, dpi=80)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig

def scatter_plot(
    x1,
    x2,
    x3=None,
    colormap=plt.cm.binary,
    color=None,
    size=None,
    hl_fc=None,
    hl_alpha=1.0,
    hl_size=10,
    title="",
    highlight_compounds=None,
    savename=None,
    to_extract=None,
):
    fig = plt.figure()
    if x3 == None:
        ax = plt.axes()
        cax = ax.scatter(x1, x2, s=size, c=color, cmap=colormap, lw=0.5)
        distmat = None
        if highlight_compounds != None:
            cmpd_hl_xy = {}
            hl_list = list(highlight_compounds)
            hl_list_found = []
            for hl in hl_list:
                if hl in to_extract["CMPD_NUMBER"].values:
                    hl_list_found.append(hl)
                    ind = pd.Index(to_extract["CMPD_NUMBER"]).get_loc(hl)
                    # print ind, hl
                    xy = (x1[ind], x2[ind])

                    cmpd_hl_xy[hl] = xy

                    colnorm = Normalize(vmin=0, vmax=best_k - 1)
                    hl_color = colormap(colnorm(hl_fc[ind]))

                    ax.annotate(
                        hl,
                        xy=xy,
                        xytext=(-20, 20),
                        weight="normal",
                        size=hl_size,
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.2", fc=hl_color, alpha=hl_alpha
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.5",
                            color="black",
                        ),
                    )

            distmat = np.zeros((len(hl_list_found), len(hl_list_found)))
            for i in range(len(hl_list_found)):
                xy_i = cmpd_hl_xy[hl_list_found[i]]
                for j in range(len(hl_list_found)):
                    if i < j:
                        xy_j = cmpd_hl_xy[hl_list_found[j]]
                        dist = math.sqrt(
                            (xy_i[0] - xy_j[0]) ** 2 + (xy_i[1] - xy_j[1]) ** 2
                        )
                        distmat[i, j] = dist
                        distmat[j, i] = dist
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        if savename != None:
            plt.savefig(savename, dpi=200)
        plt.colorbar(cax)
        plt.show()
        return distmat
    else:

        ax = Axes3D(fig)
        cax = ax.scatter(x1, x2, x3, s=s, c=c, cmap=colormap, lw=0.5)
        ax.set_zlabel("ax3")
        if savename != None:
            plt.savefig(savename, dpi=200)
        plt.colorbar(cax)
        plt.show()
        return None