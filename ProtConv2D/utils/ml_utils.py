import itertools
import math
import os
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.decomposition import (
    NMF,
    PCA,
    FactorAnalysis,
    FastICA,
    IncrementalPCA,
    KernelPCA,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder


def seq_from_onehot(x, chars_to_remove="-"):
    chars = "-ACDEFGHIKLMNPQRSTVWYX"
    a = x.argmax(axis=1)

    max_i_chars = [chars[i] for i in a if chars[i] not in chars_to_remove]

    return "".join(max_i_chars)


def add_compound_fingerprints(
    df, smiles, fptype=1, nBits=64, radius=2, col_prefix="FP"
):
    from rdkit import Chem
    from rdkit.Chem.Fingerprints import FingerprintMols
    from rdkit.Chem import MACCSkeys, AllChem
    from rdkit.Chem.MolDb import FingerprintUtils

    # smiles = to_extract["SMILES"].values
    molecules = [
        Chem.MolFromSmiles(s) for s in smiles
    ]  # pd.Series(comp_data["SMILES"].apply(Chem.MolFromSmiles))
    fail_i = []
    for i in range(len(molecules)):
        if molecules[i] == None:
            fail_i.append(i)
    if len(fail_i) > 0:
        print("Failed SMILE conversions by RDKIT:", fail_i)
    if fail_i != []:
        df = df.drop(df.index[fail_i])
    molecules = [m for m in molecules if m != None]

    assert not None in molecules

    fp_arrays = []
    fingerprint = None
    for m in molecules:
        if fptype == 0:
            fingerprint = FingerprintMols.FingerprintMol(m)
        elif fptype == 1:
            fingerprint = MACCSkeys.GenMACCSKeys(m).ToBitString()
        elif fptype == 2:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits)
        elif fptype == 3:
            fingerprint = FingerprintUtils.BuildAtomPairFP(m)  # not working!
        else:
            print("Unknown fingerprint type!")
            sys.exit(1)
        # print fingerprint
        fp_arrays.append(np.array(map(int, fingerprint)))
    fprint_length = len(fp_arrays[0])
    print("Fingerprint length:", fprint_length)
    columns = ["f%i" % i for i in range(fprint_length)]
    topological_fingerprints = pd.DataFrame(data=fp_arrays, columns=columns)

    # fprint_range = range(len(df.columns),len(df.columns)+fprint_length)
    return df.join(topological_fingerprints)


def compareDimRed(
    df,
    n_components=2,
    n_neighbors=10,
    kmin=2,
    kmax=10,
    cluster_algo=KMeans,
    color_by=None,
):
    # comparing dimensionality reduction methods in sklearn
    assert cluster_algo in [
        KMeans,
        AffinityPropagation,
        MeanShift,
        SpectralClustering,
        AgglomerativeClustering,
        DBSCAN,
        Birch,
    ]
    print("\nClustering method:", str(cluster_algo))

    color = [0 for i in range(df.shape[0])]

    nrows = 5
    ncols = 4

    print(
        "Warning: random seeds are used in all methods. Results will differ between re-runs! Otherwise, set 'random_state' argument."
    )

    print("n_neighbors (where applicable):", n_neighbors)
    models = {  #'LLE':LocallyLinearEmbedding(n_neighbors, n_components,
        #                                  eigen_solver='auto',
        #                                  method="standard"),
        #'LTSA':LocallyLinearEmbedding(n_neighbors, n_components,
        #                                  eigen_solver='dense',
        #                                  method="ltsa"),
        #'Hessian LLE':LocallyLinearEmbedding(n_neighbors, n_components,
        #                                  eigen_solver='dense',
        #                                  method="hessian"),
        #'Modified LLE':LocallyLinearEmbedding(n_neighbors, n_components,
        #                                  eigen_solver='auto',
        #                                  method="modified"),
        "Isomap": Isomap(n_neighbors=n_neighbors, n_components=n_components),
        "MDS": MDS(n_components=n_components, max_iter=100, n_init=1),
        # "SpectralEmbedding":SpectralEmbedding(n_components=n_components,n_neighbors=n_neighbors),
        "t-SNE_BarnesHut": TSNE(n_components=n_components),
        "t-SNE_Exact": TSNE(n_components=n_components, method="exact"),
        "PCA": PCA(n_components=n_components),
        # "IncrementalPCA":IncrementalPCA(n_components=n_components),
        # "ProjectedGradientNMF":ProjectedGradientNMF(n_components=n_components), # DEPRECATED
        "KernelPCA(poly)": KernelPCA(n_components=n_components, kernel="poly"),
        "KernelPCA(rbf)": KernelPCA(n_components=n_components, kernel="rbf"),
        "KernelPCA(sigmoid)": KernelPCA(n_components=n_components, kernel="sigmoid"),
        "KernelPCA(cosine)": KernelPCA(n_components=n_components, kernel="cosine"),
        # "FactorAnalysis":FactorAnalysis(n_components=n_components),
        "FastICA": FastICA(n_components=n_components),
        "TruncatedSVD": TruncatedSVD(n_components=n_components),
        # "NMF":NMF(n_components=n_components),
        "SparsePCA": SparsePCA(n_components=n_components),
    }

    fig, axarray = plt.subplots(nrows, ncols)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    if color_by != None:
        color_map = plt.cm.binary
    else:
        color_map = plt.cm.Spectral

    best_cluster_models = {}
    global_best_model = ""
    global_best_score = -1
    m = 0
    for r in range(nrows):
        for c in range(ncols):
            if m < len(models):
                mlabel = (models.keys())[m]
                print(mlabel,)
                model = models[mlabel]
                try:
                    Y = model.fit_transform(df.ix[0:, 0:])
                    x, y = Y.T

                    k_range = range(kmin, kmax + 1)
                    cluster_scores = []
                    cluster_models = []

                    # no k needed
                    if cluster_algo == AffinityPropagation:
                        damping = 0.5
                        cluster_model = cluster_algo(damping=damping).fit(Y)
                        color = cluster_model.labels_
                        print()
                    else:
                        for k in k_range:
                            # print k
                            if cluster_algo == KMeans:
                                cluster_model = cluster_algo(n_clusters=k).fit(Y)
                            elif cluster_algo == AgglomerativeClustering:
                                cluster_model = cluster_algo(n_clusters=k).fit(Y)
                            cluster_models.append(cluster_model)
                            labels = cluster_model.labels_
                            # choose most conservative scoring metric, i.e. that produces fewest number of clusters
                            # cluster_scores.append(metrics.calinski_harabaz_score(Y, labels))
                            # print len(labels) == df.shape[0]
                            score = silhouette_score(
                                Y, labels, metric="euclidean", sample_size=1000
                            )
                            # print score
                            cluster_scores.append(score)
                            # print k,score
                        assert (
                            len(k_range) == len(cluster_scores) == len(cluster_models)
                        ), (
                            "assert failed:len(k_range)=%i,  len(cluster_scores)=%i,  len(cluster_models)=%i"
                            % (len(k_range), len(cluster_scores), len(cluster_models))
                        )
                        best_score = max(cluster_scores)
                        best_k_index = cluster_scores.index(best_score)
                        best_k = k_range[best_k_index]
                        if best_score * best_k > global_best_score:
                            global_best_score = best_score * best_k
                            global_best_model = mlabel
                        if color_by == None:
                            color = cluster_models[best_k_index].labels_
                        best_cluster_models[mlabel] = (best_k, x, y, color)
                        print(
                            ": highest scoring ('Silhouette') k for clustering:", best_k
                        )
                        mlabel += "(k=%i;s=%.2f)" % (best_k, best_score)
                except Exception as err:
                    mlabel += " (ERROR!)"
                    x = np.arange(len(color)) / len(color)
                    y = np.arange(len(color)) / len(color)
                    print("ERROR:", err)
                ax = axarray[r, c]
                ax.scatter(x, y, c=color, cmap=color_map)
                ax.set_title(mlabel, fontsize=8)
                ax.set_aspect("auto")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_xlabel("dim1")
                # ax.set_ylabel("dim2")

            else:
                ax = axarray[r, c]
                fig.delaxes(ax)
            m += 1
    plt.show()
    print(
        "\nHighest-scoring DR method is %s (clustering score * num. clusters = %.2f)"
        % (global_best_model, global_best_score)
    )
    return best_cluster_models, global_best_model, global_best_score


def perform_RF(
    X,
    y,
    feat_labels,
    top=20,
    balanced_sampler_method=None,
    cv_fold=1,
    test_size=0.33,
    show_plots=False,
    n_est=None,
):
    # assert balance_sample in [None,"up","down","RandomUnderSampler","RandomOverSampler"]
    if n_est == None:
        n_est = len(feat_labels) * 2
    print("Random Forest; n=%i" % (n_est))
    ensemble_importance = {feat: 0 for feat in feat_labels}
    scores = []
    for i in range(cv_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        if balanced_sampler_method != None:
            print(float(sum(y)) / len(y))
            print(
                "Number of data points before random %s-sampling to a 50/50 class ratio (training set ONLY):"
                % str(balanced_sampler_method),
                X_train.shape[0],
            )

            X_train, y_train = sampler(balanced_sampler_method, X_train, y_train)
            X_test, y_test = sampler(balanced_sampler_method, X_test, y_test)
            #            if balance_sample=="down":
            #                X_train, y_train = balanced_subsample(X_train, y_train)
            #                X_test, y_test = balanced_subsample(X_test, y_test)
            #            elif balance_sample=="up":
            #                X_train, y_train = balanced_upsample(X_train, y_train)
            #                X_test, y_test = balanced_upsample(X_test, y_test)
            #            elif balance_sample=="RandomUnderSampler":
            #                # Create a pipeline
            #                rus = RandomUnderSampler()
            #                X_train, y_train = rus.fit_sample(X_train, y_train)
            #                X_test, y_test = rus.fit_sample(X_test, y_test)
            #                #pipeline = make_pipeline(RandomUnderSampler())
            #                #X_train, y_train = pipeline.fit(X_train, y_train)
            #                #print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
            #            else:
            #                print "ERROR: unknown sampling method!"
            #                return
            print(float(sum(y_train)) / len(y_train))
            print(
                "Number of data points after random subsampling to a 50/50 class ratio (training set ONLY):",
                y_train.shape[0],
            )

        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size
            )

        # print feat_labels

        clf = RandomForestClassifier(n_estimators=n_est)  # ,random_state=123)
        clf.fit(X_train, y_train)
        importances = clf.feature_importances_
        # print importances
        indices = np.argsort(importances)[::-1]
        # feat_labels = df.drop(["TOXIC","CMPD_NUMBER"],axis=1).columns
        # print "Feature importance (TOP %s):"%k

        for f in range(X_train.shape[1]):
            label = feat_labels[indices[f]]
            importance = importances[indices[f]]
            # print "%2d) %-*s %f" %(f+1, 30, label, importance)
            ensemble_importance[label] += importance
        print("Validation on test set:")
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        print(y_pred_proba.shape)
        # plt.scatter(y_pred_proba[0], y_test)
        # plt.show()

        if show_plots:
            fig = plot_hist(y_test, y_pred_proba.T[1], "probability", size=(3, 3))
            plt.savefig("proba_hist.pdf", figure=fig)
            plt.show()

            fig = plot_ROC(y_test, y_pred_proba.T[1], size=(3, 3))
            plt.savefig("ROC.pdf", figure=fig)
            plt.show()

        print("\nAUC: %.3f" % roc_auc_score(y_test, y_pred_proba.T[1]))
        print(confusion_matrix(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print("\nAccuracy on test data:", accuracy)
        scores.append(accuracy)

    ensemble_importance = {
        feat: ensemble_importance[feat] / cv_fold for feat in feat_labels
    }

    print("\n\nFeature importance (TOP %s):\n-----------------------------" % top)
    tmp = pd.options.display.max_colwidth

    fimps = pd.DataFrame(
        sorted(
            ensemble_importance.items(), key=lambda kv: (kv[1], kv[0]), reverse=True
        )[:top],
        columns=["feature", "mean_importance"],
    )
    pd.options.display.max_colwidth = max([len(i) for i in fimps["feature"]]) + 2
    print(fimps)
    pd.options.display.max_colwidth = tmp

    scores = np.array(scores)
    # scores = cross_val_score(clf, X,y, cv=cv_fold)
    print(
        "Accuracy (%i-fold CV): %0.2f (+/- %0.2f)"
        % (cv_fold, scores.mean(), scores.std() * 2)
    )

    # predicted = cross_val_predict(clf, X, y, cv=cv_fold)
    # print accuracy_score(y, predicted)
    return scores.mean()  # accuracy_score(y, predicted)


def perform_DNN(
    df,
    y,
    arch,
    ref_val=None,
    ref_val_label="",
    act="relu",
    dropout=0.25,
    ep=50,
    bs=50,
    test_fraction=0.5,
    balanced_sampler_method=None,
):
    # assert balance_classes in [None, "up", "down"]
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout

    # from keras.optimizers import SGD

    model = Sequential()

    X = df.ix[:, :].values

    #    if balance_classes != None:
    #        if balance_classes =="down":
    #            X,y = balanced_subsample(X,y.values)
    #        else:
    #            X,y = balanced_upsample(X,y.values)

    # y = pd.get_dummies(y).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)

    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", y_train.shape)
    if balanced_sampler_method != None:
        # print float(sum(y))/len(y)
        print(
            "Number of data points BEFORE random %s-sampling to a 50/50 class ratio (train/test): %i/%i"
            % (str(balanced_sampler_method), X_train.shape[0], X_test.shape[0])
        )

        X_train, y_train = sampler(balanced_sampler_method, X_train, y_train)
        X_test, y_test = sampler(balanced_sampler_method, X_test, y_test)
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values
        # print float(sum(y_train))/len(y_train)
        print(
            "Number of data points AFTER random %s-sampling to a 50/50 class ratio (train/test): %i/%i"
            % (str(balanced_sampler_method), X_train.shape[0], X_test.shape[0])
        )

    assert len(arch) >= 1
    model.add(Dense(output_dim=arch[0] * X_train.shape[1], input_dim=X_train.shape[1]))
    model.add(Activation(act))
    model.add(Dropout(dropout))
    for i in range(1, len(arch) - 1):
        print("hidden layer %i (%i neurons)" % (i, arch[i] * X_train.shape[1]))
        model.add(Dense(output_dim=arch[i] * X_train.shape[1]))
        model.add(Activation(act))
        model.add(Dropout(dropout))

    model.add(Dense(output_dim=y_train.shape[1]))
    model.add(Activation("softmax"))

    my_metrics = ["accuracy"]

    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=my_metrics)
    print(model)
    print(model.summary())
    history = model.fit(
        X_train,
        y_train,
        epochs=ep,
        batch_size=bs,
        verbose=True,
        validation_data=(X_test, y_test),
        shuffle=True,
    )

    # important if you want to re-use your trained model in the future!
    model.save_weights("model_weight.hdf5")

    # plot the training and validation metrics over time
    train_metric = "acc"
    test_metric = "val_" + train_metric
    d = history.history
    print(d.keys())

    plt.plot(d[train_metric], linestyle="-", color="black", label=train_metric)
    plt.plot(d[test_metric], linestyle=":", color="black", label=test_metric)
    if ref_val != None:
        plt.plot(
            [ref_val for i in range(ep)],
            linestyle="--",
            color="black",
            label=ref_val_label,
        )
    plt.xlabel("epoch")
    plt.ylabel("metric:" + train_metric)
    plt.legend(loc="lower right")
    plt.show()


def perform_rnn(df, seqs, Y, ep=3, bs=64):
    from keras.models import Sequential
    from keras.layers import (
        Dense,
        Activation,
        Dropout,
        TimeDistributed,
        RepeatVector,
        recurrent,
    )

    # from keras.optimizers import SGD
    from keras.layers.embeddings import Embedding
    from keras.preprocessing import sequence
    from keras.layers import Input, LSTM, concatenate, BatchNormalization
    from keras.models import Model

    # fix random seed for reproducibility
    # np.random.seed(7)

    # All the numbers, plus sign and space for padding.
    chars = "ACDEFGHIKLMNPQRSTVWY"
    aa2num = {chars[a]: a + 1 for a in range(len(chars))}
    # assert len(chars)==20
    # ctable = CharacterTable(chars)

    Xnum = df.values
    Xseq = np.array([None for i in range(len(seqs))])
    for i in range(len(Xseq)):

        Xseq[i] = np.array([aa2num[letter] for letter in seqs[i]])
        # print X[i]
    max_seq_length = 48

    print(np.mean([len(i) for i in Xseq]), "average seq length")
    print(np.max([len(i) for i in Xseq]), "max seq length")

    Xseq = sequence.pad_sequences(Xseq, maxlen=max_seq_length)

    # X = df.ix[:,:].values
    Y = Y.values
    # Y = pd.get_dummies(Y).values
    print(Xseq.shape)
    print(Xnum.shape)

    # Xnum_train, Xnum_test, Xseq_train, Xseq_test, y_train, y_test = train_test_split(Xnum, Xseq, Y, test_size=0.33)
    # truncate and pad input sequences

    # X_train[:,-1] = sequence.pad_sequences(X_train[:,-1], maxlen=max_seq_length)
    # X_test[:,-1] = sequence.pad_sequences(X_test[:,-1], maxlen=max_seq_length)

    # headline input: meant to receive sequences of 100 integers, between 1 and 10000.
    # note that we can name any layer by passing it a "name" argument.
    main_input = Input(shape=(max_seq_length,), dtype="int32", name="main_input")

    # this embedding layer will encode the input sequence
    # into a sequence of dense 512-dimensional vectors.
    emb = Embedding(output_dim=512, input_dim=21, input_length=max_seq_length)(
        main_input
    )

    # a LSTM will transform the vector sequence into a single vector,
    # containing information about the entire sequence
    lstm_out = LSTM(64)(emb)

    auxiliary_input = Input(shape=(9,), name="aux_input")
    x = Dense(64)(auxiliary_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    aux = Dropout(0.25)(x)
    # auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    print(lstm_out.shape, auxiliary_input.shape)

    x = concatenate([lstm_out, aux], axis=1)
    print(x.shape)
    # we stack a deep fully-connected network on top

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.25)(x)

    # and finally we add the main logistic regression layer
    main_output = Dense(1, activation="sigmoid", name="main_output")(x)
    model = Model(input=[main_input, auxiliary_input], output=[main_output])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy',
    #          loss_weights=[1., 0.2])
    # model.fit([, additional_data], [labels, labels],
    #      nb_epoch=50, batch_size=32)
    my_metrics = ["accuracy"]
    model.compile(
        optimizer="adam",
        loss={"main_output": "binary_crossentropy"},
        loss_weights={"main_output": 1.0},
        metrics=my_metrics,
    )
    print(model)
    print(model.summary())
    # and trained it via:
    history = model.fit(
        {"main_input": Xseq, "aux_input": Xnum},
        {"main_output": Y, "aux_output": Y},
        validation_split=0.33,
        epochs=ep,
        batch_size=bs,
    )

    # create the model
    #    embedding_vecor_length = 32
    #
    #    model = Sequential()
    #    model.add(Embedding(21, embedding_vecor_length, input_length=max_seq_length))
    #    model.add(recurrent.LSTM(64))
    #    #model.add(recurrent.LSTM(64))
    #    model.add(Dense(32, activation='relu'))
    #    #model.add(Dense(32, activation='relu'))
    #    model.add(Dense(1, activation='sigmoid'))
    #    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #    print(model.summary())
    #    #model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
    #
    #
    #    my_metrics = ['accuracy','fbeta_score']
    #
    #    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=my_metrics)
    #
    #
    #    history = model.fit(X_train, y_train,
    #                        nb_epoch=ep,
    #                        batch_size=bs,
    #                        verbose=True,
    #                        validation_data=(X_test, y_test),
    #                        shuffle=True)

    # important if you want to re-use your trained model in the future!
    modlabel = "test_lstm_merge_decoy"
    print(history.history)
    with open("%s_model_history.json" % modlabel, "w") as hist:
        hist.write(str(history.history))
    model.save_weights("%s_model_weights.hdf5" % modlabel)
    with open("%s_model_arch.json" % modlabel, "w") as json:
        json.write(model.to_json())

    # plot the training and validation metrics over time
    train_metric = "acc"
    test_metric = "val_" + train_metric
    d = history.history
    print(d.keys())

    plt.plot(d[train_metric], linestyle="-", color="black", label=train_metric)
    plt.plot(d[test_metric], linestyle=":", color="black", label=test_metric)
    # if ref_val != None: plt.plot( [ref_val for i in range(ep)], linestyle="--", color="black", label=ref_val_label )
    plt.xlabel("epoch")
    plt.ylabel("metric:" + train_metric)
    plt.legend(loc="lower right")
    plt.savefig("rnn_acc.png")
    plt.show()


def perform_cnn_2d(
    X,
    Y,
    dim=32,
    batch_size=32,
    nb_epoch=100,
    generic_label="test",
    data_augmentation=False,
    data_limit=None,
    load_model=False,
    dimord="th",
    RF_baseline=True,
    img_channels=1,
    ker_rows=3,
    ker_cols=3,
    ref_val=None,
):
    from keras.utils import np_utils
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential, model_from_json
    from keras.layers import (
        Dense,
        Dropout,
        Activation,
        Flatten,
        Convolution2D,
        MaxPooling2D,
    )

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    nb_classes = len(pd.Series(Y).unique())
    print(nb_classes, "classes")

    modlabel = "%s_cnn_%ix%ix%i_bt%i_ep%i_f%ix%i" % (
        generic_label,
        dim,
        dim,
        img_channels,
        batch_size,
        nb_epoch,
        ker_rows,
        ker_cols,
    )

    print("CNN:")
    # The data, shuffled and split between train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0
    )

    print("X_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print(X_test.shape[0], "test samples")
    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    if not load_model:
        model = Sequential()

        model.add(
            Convolution2D(
                32,
                ker_rows,
                ker_cols,
                border_mode="same",
                input_shape=X_train.shape[1:],
                dim_ordering=dimord,
            )
        )
        model.add(Activation("relu"))
        model.add(Convolution2D(32, ker_rows, ker_cols))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, ker_rows, ker_cols, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, ker_rows, ker_cols))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation("softmax"))
    else:
        with open("model_arch.json") as arch:
            model = model_from_json(arch.read())
        model.load_weights("model_weights.hdf5")

    # Let's train the model using RMSprop
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy", "fbeta_score"],
    )

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    print("Training this model:")
    print(model.summary())
    # print model.get_config()

    if not data_augmentation:
        print("Not using data augmentation.")
        history = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            validation_data=(X_test, Y_test),
            shuffle=True,
        )
    else:
        print("Using real-time data augmentation.")
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            dim_ordering=dimord,
        )
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch,
            validation_data=(X_test, Y_test),
        )
    print(history.history)
    with open("%s_model_history.json" % modlabel, "w") as hist:
        hist.write(str(history.history))
    model.save_weights("%s_model_weights.hdf5" % modlabel)
    with open("%s_model_arch.json" % modlabel, "w") as json:
        json.write(model.to_json())

class DummySampler(object):
    def sample(self, X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


