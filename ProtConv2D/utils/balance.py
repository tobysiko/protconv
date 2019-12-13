def compareSamplingMethods(X, y):
    # TODO unfinished!!

    cv = StratifiedKFold(n_splits=3)
    RANDOM_STATE = 42  # obviously
    LW = 2

    # classifier = ['3NN', neighbors.KNeighborsClassifier(3)]
    classifier = ["RF", RandomForestClassifier(n_estimators=100)]
    samplers = [
        ["Standard", MLhelpers.DummySampler()],
        ["ADASYN", ADASYN(random_state=RANDOM_STATE)],
        ["ROS", RandomOverSampler(random_state=RANDOM_STATE)],
        ["SMOTE", SMOTE(random_state=RANDOM_STATE)],
    ]

    pipelines = [
        [
            "{}-{}".format(sampler[0], classifier[0]),
            make_pipeline(sampler[1], classifier[1]),
        ]
        for sampler in samplers
    ]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for name, pipeline in pipelines:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, y):
            probas_ = pipeline.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)

        mean_tpr /= cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr,
            mean_tpr,
            linestyle="--",
            label="{} (area = %0.2f)".format(name) % mean_auc,
            lw=LW,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=LW, color="k", label="Luck")

    # make nice plotting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")

    plt.legend(loc="lower right")

    plt.show()

def sampler(method, X, y, n_jobs=2):
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import (
        RandomUnderSampler,
        ClusterCentroids,
        NearMiss,
        AllKNN,
        EditedNearestNeighbours,
        CondensedNearestNeighbour,
        RepeatedEditedNearestNeighbours,
    )

    # UNDERSAMPLING
    if method == "RandomUnderSampler":
        sampler = RandomUnderSampler()
    elif method == "ClusterCentroids":
        sampler = ClusterCentroids()
    elif method == "NearMiss":
        sampler = NearMiss()
    elif method == "AllKNN":
        sampler = AllKNN(n_jobs=n_jobs)
    elif method == "EditedNearestNeighbours":
        sampler = EditedNearestNeighbours()
    elif method == "CondensedNearestNeighbour":
        sampler = CondensedNearestNeighbour()
    elif method == "RepeatedEditedNearestNeighbours":
        sampler = RepeatedEditedNearestNeighbours()
    # OVERSAMPLING
    elif method == "SMOTE":
        sampler = SMOTE(n_jobs=n_jobs)
    elif method == "ADASYN":
        sampler = ADASYN(n_jobs=n_jobs)
    else:
        return X, y
    return sampler.fit_sample(X, y)

# y is sinle-column target class vector
def balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    # iterate over target classes
    for yi in np.unique(y):
        # select elements in x is in class yi
        elems = x[(y == yi)]
        # store x-elements for each class yi
        class_xs.append((yi, elems))
        # find yi with smallest number of members --> min_elems
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    print("Down-sampling to smallest number of elements in a class:", min_elems)
    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    # iterate of yi,x(yi) pairs
    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    print(xs.shape, ys.shape)
    return xs, ys


# y is single-column target class vector
def balanced_upsample(x, y, upsample_factor=1.0, noise_factor=0.0):

    class_xs = []
    max_elems = None

    # iterate over target classes
    for yi in np.unique(y):
        # select elements in x is in class yi
        elems = x[(y == yi)]
        # store x-elements for each class yi
        class_xs.append((yi, elems))
        # find yi with smallest number of members --> min_elems
        if max_elems == None or elems.shape[0] > max_elems:
            max_elems = elems.shape[0]

    print("Up-sampling to largest number of elements in a class:", max_elems)
    use_elems = max_elems
    if upsample_factor != 1.0:
        use_elems = int(max_elems * upsample_factor)

    xs = []
    ys = []

    # iterate of yi,x(yi) pairs
    for ci, this_xs in class_xs:
        irange = list(range(len(this_xs)))
        if len(this_xs) < use_elems:

            ichoice = np.random.choice(irange, use_elems, replace=True)

        else:
            ichoice = irange  # np.random.shuffle(irange)

        x_ = this_xs[ichoice]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    print(xs.shape, ys.shape)
    return xs, ys