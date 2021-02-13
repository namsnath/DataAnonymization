 	#!/usr/bin/env python
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches
from tabulate import tabulate
import os
import warnings
import sys

# if sys.platform != "win32":
#     import pymp

warnings.filterwarnings("ignore")

def anonymize(k, l, t, verbose=False, plots=False, write_files=False):
    K_VALUE = k
    L_VALUE = l
    T_VALUE = t
    DISPLAY_VERBOSE = verbose
    DISPLAY_PLOTS = plots

    data_headers = (
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    )

    categorical_variables = set(
        (
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "sex",
            "native-country",
            "race",
            "income",
        )
    )


    def get_spans(dataFrame, partition, scale=None):
        spans = {}
        for column in dataFrame.columns:
            if column in categorical_variables:
                span = len(dataFrame[column][partition].unique())
            else:
                span = dataFrame[column][partition].max()
                -dataFrame[column][partition].min()
            if scale is not None:
                span = span / scale[column]
            spans[column] = span
        return spans


    def split(dataFrame, partition, column):
        dfp = dataFrame[column][partition]
        if column in categorical_variables:
            values = dfp.unique()
            lv = set(values[: len(values) // 2])
            rv = set(values[len(values) // 2 :])
            return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
        else:
            median = dfp.median()
            dfl = dfp.index[dfp < median]
            dfr = dfp.index[dfp >= median]
            return (dfl, dfr)


    def is_k_anonymous(dataFrame, partition, sensitive_column, k=K_VALUE):
        if len(partition) < k:
            return False
        return True


    def partition_dataset(dataFrame, feature_columns, sensitive_column, scale, is_valid):
        finished_partitions = []
        partitions = [dataFrame.index]
        while partitions:
            partition = partitions.pop(0)
            spans = get_spans(dataFrame[feature_columns], partition, scale)
            for column, span in sorted(spans.items(), key=lambda x: -x[1]):
                lp, rp = split(dataFrame, partition, column)
                if not is_valid(dataFrame, lp, sensitive_column) or not is_valid(
                    dataFrame, rp, sensitive_column
                ):
                    continue
                partitions.extend((lp, rp))
                break
            else:
                finished_partitions.append(partition)
        return finished_partitions


    def build_indexes(dataFrame):
        indexes = {}
        for column in categorical_variables:
            values = sorted(dataFrame[column].unique())
            indexes[column] = {x: y for x, y in zip(values, range(len(values)))}
        return indexes


    def get_coords(dataFrame, column, partition, indexes, offset=0.1):
        if column in categorical_variables:
            sv = dataFrame[column][partition].sort_values()
            l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]] + 1.0
        else:
            sv = dataFrame[column][partition].sort_values()
            next_value = sv[sv.index[-1]]
            larger_values = dataFrame[dataFrame[column] > next_value][column]
            if len(larger_values) > 0:
                next_value = larger_values.min()
            l = sv[sv.index[0]]
            r = next_value
        l -= offset
        r += offset
        return l, r


    def get_partition_rects(
        dataFrame, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]
    ):
        rects = []
        for partition in partitions:
            xl, xr = get_coords(dataFrame, column_x, partition, indexes, offset=offsets[0])
            yl, yr = get_coords(dataFrame, column_y, partition, indexes, offset=offsets[1])
            rects.append(((xl, yl), (xr, yr)))
        return rects


    def get_bounds(dataFrame, column, indexes, offset=1.0):
        if column in categorical_variables:
            return 0 - offset, len(indexes[column]) + offset
        return dataFrame[column].min() - offset, dataFrame[column].max() + offset


    def plot_rects(
        dataFrame, ax, rects, column_x, column_y, indexes, edgecolor="black", facecolor="none"
    ):
        for (xl, yl), (xr, yr) in rects:
            ax.add_patch(
                patches.Rectangle(
                    (xl, yl),
                    xr - xl,
                    yr - yl,
                    linewidth=1,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    alpha=0.5,
                )
            )
        ax.set_xlim(*get_bounds(dataFrame, column_x, indexes))
        ax.set_ylim(*get_bounds(dataFrame, column_y, indexes))
        ax.set_xlabel(column_x)
        ax.set_ylabel(column_y)


    def agg_categorical_column(series):
        return [",".join(set(series))]


    def agg_numerical_column(series):
        return [series.mean()]


    def build_anonymized_dataset(
        dataFrame, partitions, feature_columns, sensitive_column, max_partitions=None
    ):
        aggregations = {}
        for column in feature_columns:
            if column in categorical_variables:
                aggregations[column] = agg_categorical_column
            else:
                aggregations[column] = agg_numerical_column
        rows = []
        for i, partition in enumerate(partitions):
            # if i % 100 == 1:
            #     print("Finished {} partitions...".format(i))
            if max_partitions is not None and i > max_partitions:
                break
            grouped_columns = dataFrame.loc[partition].agg(aggregations, squeeze=False)
            sensitive_counts = (
                dataFrame.loc[partition]
                .groupby(sensitive_column)
                .agg({sensitive_column: "count"})
            )
            values = grouped_columns.iloc[0].to_dict()
            for sensitive_value, count in sensitive_counts[sensitive_column].items():
                if count == 0:
                    continue
                values.update(
                    {sensitive_column: sensitive_value, "count": count,}
                )
                rows.append(values.copy())
        return pd.DataFrame(rows)


    def diversity(dataFrame, partition, column):
        return len(dataFrame[column][partition].unique())


    def is_l_diverse(dataFrame, partition, sensitive_column, l=L_VALUE):
        return diversity(dataFrame, partition, sensitive_column) >= l


    def t_closeness(dataFrame, partition, column, global_freqs):
        total_count = float(len(partition))
        d_max = None
        group_counts = dataFrame.loc[partition].groupby(column)[column].agg("count")
        for value, count in group_counts.to_dict().items():
            p = count / total_count
            d = abs(p - global_freqs[value])
            if d_max is None or d > d_max:
                d_max = d
        return d_max


    def is_t_close(dataFrame, partition, sensitive_column, global_freqs, p=T_VALUE):
        if not sensitive_column in categorical_variables:
            raise ValueError("this method only works for categorial values")
        return t_closeness(dataFrame, partition, sensitive_column, global_freqs) <= p


    def calc_diff(orig_val, new_val, round_to=-1):
        val = ((new_val - orig_val) / orig_val * 100)

        if round_to == -1:
            return val

        return round(val, round_to)


##################################################################################################

    # Read the CSV into a Pandas Dataframe
    dataFrame = pd.read_csv(
        "./data/adult.all.txt",
        sep=", ",
        header=None,
        names=data_headers,
        index_col=False,
        engine="python",
    )

    # Assign type to catgorical variables
    for header in categorical_variables:
        dataFrame[header] = dataFrame[header].astype("category")

    if DISPLAY_VERBOSE:
        print("Data Preview:")
        print(dataFrame.head())
        print("\n\n")

    full_spans = get_spans(dataFrame, dataFrame.index)
    # print('Spans: ', full_spans)
    # print('\n\n')

    feature_columns = ["age", "education-num"]
    sensitive_column = "income"

    # K-ANONYMITY

    finished_partitions = partition_dataset(
        dataFrame, feature_columns, sensitive_column, full_spans, is_k_anonymous
    )

    indexes = build_indexes(dataFrame)
    column_x, column_y = feature_columns[:2]
    rects = get_partition_rects(
        dataFrame, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0]
    )

    # print(rects[:10])

    pl.figure(1, figsize=(20, 20))
    ax = pl.subplot(1, 1, 1)
    plot_rects(dataFrame, ax, rects, column_x, column_y, indexes=indexes, edgecolor="black")
    pl.scatter(dataFrame[column_x], dataFrame[column_y])

    k_anonymous_dataframe = build_anonymized_dataset(
        dataFrame, finished_partitions, feature_columns, sensitive_column
    )

    # L-DIVERSITY

    finished_l_diverse_partitions = partition_dataset(
        dataFrame,
        feature_columns,
        sensitive_column,
        full_spans,
        lambda *args: is_k_anonymous(*args) and is_l_diverse(*args),
    )

    column_x, column_y = feature_columns[:2]
    l_diverse_rects = get_partition_rects(
        dataFrame,
        finished_l_diverse_partitions,
        column_x,
        column_y,
        indexes,
        offsets=[0.0, 0.0],
    )

    pl.figure(2, figsize=(20, 20))
    ax = pl.subplot(1, 1, 1)
    plot_rects(dataFrame, ax, l_diverse_rects, column_x, column_y, indexes=indexes, edgecolor="black")
    pl.scatter(dataFrame[column_x], dataFrame[column_y])

    l_diverse_dataframe = build_anonymized_dataset(
        dataFrame, finished_l_diverse_partitions, feature_columns, sensitive_column
    )

    # T-CLOSENESS

    global_freqs = {}
    total_count = float(len(dataFrame))
    group_counts = dataFrame.groupby(sensitive_column)[sensitive_column].agg("count")
    for value, count in group_counts.to_dict().items():
        p = count / total_count
        global_freqs[value] = p

    # print('Global Frequencies: ')
    # print(global_freqs)
    # print('\n\n')


    finished_t_close_partitions = partition_dataset(
        dataFrame,
        feature_columns,
        sensitive_column,
        full_spans,
        lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs),
    )

    t_close_dataframe = build_anonymized_dataset(
        dataFrame, finished_t_close_partitions, feature_columns, sensitive_column
    )      
    

    column_x, column_y = feature_columns[:2]
    t_close_rects = get_partition_rects(
        dataFrame,
        finished_t_close_partitions,
        column_x,
        column_y,
        indexes,
        offsets=[0.0, 0.0],
    )

    pl.figure(3, figsize=(20, 20))
    ax = pl.subplot(1, 1, 1)
    plot_rects(dataFrame, ax, t_close_rects, column_x, column_y, indexes=indexes, edgecolor="black")
    pl.scatter(dataFrame[column_x], dataFrame[column_y])
        



    if DISPLAY_VERBOSE:
        print("K-Anon Partition Count: ", len(finished_partitions))
        print("K-Anon Dataframe:")
        print(k_anonymous_dataframe.sort_values(feature_columns + [sensitive_column]))
        print("\n\n")

        print("L-Diverse Partition Count: ", len(finished_l_diverse_partitions))
        print("L-Diverse Dataframe: ")
        print(l_diverse_dataframe.sort_values([column_x, column_y, sensitive_column]))
        print("\n\n")

        print("T-Close Partition Count: ", len(finished_t_close_partitions))
        print("T-Close Dataframe: ")
        print(t_close_dataframe.sort_values([column_x, column_y, sensitive_column]))
        print("\n\n")

    if DISPLAY_PLOTS:
        pl.show()


    means = {
        "original": (dataFrame["age"].mean(), dataFrame["education-num"].mean()), 
    "original": (dataFrame["age"].mean(), dataFrame["education-num"].mean()), 
        "original": (dataFrame["age"].mean(), dataFrame["education-num"].mean()), 
    "original": (dataFrame["age"].mean(), dataFrame["education-num"].mean()), 
        "original": (dataFrame["age"].mean(), dataFrame["education-num"].mean()), 
        "k": k_anonymous_dataframe.mean(),
        "l": l_diverse_dataframe.mean(),
        "t": t_close_dataframe.mean(),
    }

    sd = {
        "original": (dataFrame["age"].std(), dataFrame["education-num"].std()),
        "k": k_anonymous_dataframe.std(),
        "l": l_diverse_dataframe.std(),
        "t": t_close_dataframe.std(),
    }

    diffs = {
        "means": {
            "age": {
                "k": calc_diff(means["original"][0], means["k"][0]),
                "l": calc_diff(means["original"][0], means["l"][0]),
                "t": calc_diff(means["original"][0], means["t"][0]),
            },
            "edu": {
                "k": calc_diff(means["original"][1], means["k"][1]),
                "l": calc_diff(means["original"][1], means["l"][1]),
                "t": calc_diff(means["original"][1], means["t"][1]),
            }
        },
        "sd": {
        "age": {
                "k": calc_diff(sd["original"][0], sd["k"][0]),
                "l": calc_diff(sd["original"][0], sd["l"][0]),
                "t": calc_diff(sd["original"][0], sd["t"][0]),
            },
            "edu": {
                "k": calc_diff(sd["original"][1], sd["k"][1]),
                "l": calc_diff(sd["original"][1], sd["l"][1]),
                "t": calc_diff(sd["original"][1], sd["t"][1]),
            }
        }
    }

    table_age = [
        ["Algorithm", "Mean", "Mean Diff %", "SD", "SD Diff %"],
        ["Original", means["original"][0], "", sd["original"][0], ""],
        ["K-Anonymity", means["k"][0], diffs["means"]["age"]["k"], sd["k"][0], diffs["sd"]["age"]["k"]],
        ["L-Diversity", means["l"][0], diffs["means"]["age"]["l"], sd["l"][0], diffs["sd"]["age"]["l"]],
        ["T-Closeness", means["t"][0], diffs["means"]["age"]["t"], sd["t"][0], diffs["sd"]["age"]["t"]]
    ]

    table_edu = [
        ["Algorithm", "Mean", "Mean Diff %", "SD", "SD Diff %"],
        ["Original", means["original"][1], "", sd["original"][1], ""],
        ["K-Anonymity", means["k"][1], diffs["means"]["edu"]["k"], sd["k"][1], diffs["sd"]["edu"]["k"]],
        ["L-Diversity", means["l"][1], diffs["means"]["edu"]["l"], sd["l"][1], diffs["sd"]["edu"]["l"]],
        ["T-Closeness", means["t"][1], diffs["means"]["edu"]["t"], sd["t"][1], diffs["sd"]["edu"]["t"]]
    ]

    table_part_count = [
        ["Algorithm", "Value", "Partition Count"],
        ["K-Anonymity", K_VALUE, len(finished_partitions)],
        ["L-Diversity", L_VALUE, len(finished_l_diverse_partitions)],
        ["T-Closeness", T_VALUE, len(finished_t_close_partitions)]
    ]

    if DISPLAY_VERBOSE:
        print("\nPartitions:")
        print(tabulate(table_part_count, headers="firstrow"))

        print("\nAge:")
        print(tabulate(table_age, headers="firstrow"))

        print("\nEducation:")
        print(tabulate(table_edu, headers="firstrow"))

    output_csv = "K-Anonymity,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\nL-Diversity,{10},{11},{12},{13},{14},{15},{16},{17},{18},{19}\nT-Closeness,{20},{21},{22},{23},{24},{25},{26},{27},{28},{29}\n".format(
        K_VALUE, len(finished_partitions), 
        means["k"][0], diffs["means"]["age"]["k"], means["k"][1], diffs["means"]["edu"]["k"],
        sd["k"][0], diffs["sd"]["age"]["k"], sd["k"][1], diffs["sd"]["edu"]["k"],

        L_VALUE, len(finished_l_diverse_partitions), 
        means["l"][0], diffs["means"]["age"]["l"], means["l"][1], diffs["means"]["edu"]["l"],
        sd["l"][0], diffs["sd"]["age"]["l"], sd["l"][1], diffs["sd"]["edu"]["l"],

        T_VALUE, len(finished_t_close_partitions), 
        means["t"][0], diffs["means"]["age"]["t"], means["t"][1], diffs["means"]["edu"]["t"],
        sd["t"][0], diffs["sd"]["age"]["t"], sd["t"][1], diffs["sd"]["edu"]["t"]
    )

    if write_files:
        k_anonymous_dataframe.to_csv("k-anon.csv")
        l_diverse_dataframe.to_csv("l-diverse.csv")
        t_close_dataframe.to_csv("t-close.csv")

    return output_csv

def main():
    # k_values = [3, 5, 7, 10]
    # l_values = [2]
    # t_values = [0.15, 0.2, 0.3, 0.4, 0.5]

    k_values = [10]
    l_values = [2]
    t_values = [0.4]

    full_csv = ""

    if sys.platform == "win32":
        for k in k_values:
            for l in l_values:
                for t in t_values:
                    print(k, l, t)
                    csv = anonymize(k, l, t)
                    full_csv = full_csv + csv
    else:
        csv_data = pymp.shared.dict()
        pymp.config.nested = True
        for k in k_values:
            for l in l_values:
                with pymp.Parallel(6) as p:
                    for t in p.iterate(t_values):
                        print(k, l, t)
                        csv = anonymize(k, l, t)
                        csv_data["K={0}, L={1}, T={2}".format(k, l, t)] = csv
        full_csv = "".join([v for k,v in csv_data.items()])

    with open(os.path.join(os.path.curdir, "Result.csv"), "w") as f:
        f.write(full_csv)

if __name__ == "__main__":
    # main()
    csv = anonymize(10, 2, 0.4, plots=True)
    print(csv)