# base imputation of each row on random subset of table
def randomized_imputation(
    df, batch=1, method="mean", impute_fraction=0.7, verbose=False
):
    if verbose:
        print_progress(
            0, df.shape[0], prefix="Imputing:", suffix="", decimals=1, bar_length=50
        )

    imp = Imputer(missing_values="NaN", strategy=method, axis=1)
    for i in df.index.values:
        if i % batch == 0:
            if verbose:
                print_progress(
                    i + 1,
                    df.shape[0],
                    prefix="Imputing:",
                    suffix="",
                    decimals=1,
                    bar_length=50,
                )
            sample = np.random.choice(
                df.index.values, int(df.shape[0] * impute_fraction)
            )
            imp = imp.fit(df.ix[sample, :])
            df.ix[i : i + batch - 1, :] = imp.transform(df.ix[i : i + batch - 1, :])

    if verbose:
        print_progress(
            i + 1, df.shape[0], prefix="Imputing:", suffix="", decimals=1, bar_length=50
        )
        print
    return df