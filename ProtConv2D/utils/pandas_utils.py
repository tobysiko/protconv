def fracNArows(df, f):
    if f > 0:  # remove rows
        print(df.shape, int(f * df.shape[1]), " non-NA required in row")
        df.dropna(axis=0, thresh=int(f * df.shape[1]), inplace=True)


def fracNAcols(df, f):
    if f > 0:  # remove cols
        print(df.shape, int(f * df.shape[0]), " non-NA required in column")
        df.dropna(axis=1, thresh=int(f * df.shape[0]), inplace=True)


def addOnData(
    df,
    data_list=[],
    merge_on=None,
    extract={},
    exclude=[],
    row_filled_fraction=0.0,
    col_filled_fraction=0.0,
    rowsFirst=True,
    verbose=True,
):
    if verbose:
        print("Data frame shape:", df.shape)
    for di in range(len(data_list)):
        d = data_list[di]

        suffix = os.path.splitext(d)[1]
        if suffix == ".csv":
            table = pd.read_csv(d)
        elif suffix == ".xlsx":
            table = pd.read_excel(d)
        else:
            table = pd.read_table(d)

        df = pd.merge(df, table, on=merge_on)
        if verbose:
            print("Shape:", df.shape, "after merging", d)

    if rowsFirst:
        fracNArows(df, row_filled_fraction)
        fracNAcols(df, col_filled_fraction)
    else:
        fracNAcols(df, col_filled_fraction)
        fracNArows(df, row_filled_fraction)

    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    if verbose:
        print("DF shape after removing missing values:", df.shape)
    # remove these columns
    # If a '*' is found in the column name and no exact match is found, a substring search will be performed instead, deleting  all columns containing the substring.
    for col in exclude:
        if not col in df.columns.values:
            if "*" in col:
                for col2 in df.columns.values:
                    if col.strip("*") in col2:
                        del df[col2]
        else:
            del df[col]
    # keep these columns in separate arrays.
    # IMPORTANT: make sure that no rows are removed after this step - unless you remove the same rows in the excluded columns
    for col in extract:
        if col in df.columns:
            # coltype = extract[col]
            extract[col] = None
            extract[col] = df[col]

            del df[col]
    # na_mask = df==np.nan

    # print na_mask, na_mask.shape, na_mask.describe()
    return df

def scaleColumns(df, method, cols_to_scale):
    df[cols_to_scale] = method.fit_transform(df[cols_to_scale])
    return df


def shuffleDFrows(df):
    return df.reindex(np.random.permutation(df.index))


def DFcounts(df, drop_1counts=False):
    # Print column names, types and unique count
    todrop = []
    col_summary = []
    for c in df.columns:
        n = len(df[c].unique())

        coltype = df[c].dtype

        colmin = None
        colmax = None
        nacount = df[c].isnull().sum()
        if coltype in [int, float, np.int32, np.float32, np.bool, np.int64, np.float64]:
            colmin = df[c].min()
            colmax = df[c].max()

        if n <= 1:
            todrop.append(c)
        col_summary.append([c, coltype, n, nacount, colmin, colmax])

    if drop_1counts:
        print("\nDropping uninformative columns:")
        for c in todrop:
            print("\t%s - only value: %s" % (c, df[c].unique()))
            del df[c]

    if col_summary != []:
        return pd.DataFrame(
            col_summary, columns=["Column", "Type", "#unique", "#NA", "min", "max"]
        )
    else:
        return None