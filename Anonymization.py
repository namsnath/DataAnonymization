import pandas as pd
import matplotlib.pylab as pl
import matplotlib.patches as patches

data_headers = (
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
)

categorical_variables = set((
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'sex',
    'native-country',
    'race',
    'income',
))


def get_spans(dataFrame, partition, scale=None):
    spans = {}
    for column in dataFrame.columns:
        if column in categorical_variables:
            span = len(dataFrame[column][partition].unique())
        else:
            span = dataFrame[column][partition].max()
            - dataFrame[column][partition].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def split(dataFrame, partition, column):
    dfp = dataFrame[column][partition]
    if column in categorical_variables:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


def is_k_anonymous(dataFrame, partition, sensitive_column, k=3):
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
            if not is_valid(dataFrame, lp, sensitive_column) or not is_valid(dataFrame, rp, sensitive_column):
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
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
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


def get_partition_rects(dataFrame, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        xl, xr = get_coords(dataFrame, column_x, partition,
                            indexes, offset=offsets[0])
        yl, yr = get_coords(dataFrame, column_y, partition,
                            indexes, offset=offsets[1])
        rects.append(((xl, yl), (xr, yr)))
    return rects


def get_bounds(dataFrame, column, indexes, offset=1.0):
    if column in categorical_variables:
        return 0-offset, len(indexes[column])+offset
    return dataFrame[column].min()-offset, dataFrame[column].max()+offset


def plot_rects(dataFrame, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
    for (xl, yl), (xr, yr) in rects:
        ax.add_patch(patches.Rectangle((xl, yl), xr-xl, yr-yl, linewidth=1,
                                       edgecolor=edgecolor, facecolor=facecolor, alpha=0.5))
    ax.set_xlim(*get_bounds(dataFrame, column_x, indexes))
    ax.set_ylim(*get_bounds(dataFrame, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)


def agg_categorical_column(series):
    return [','.join(set(series))]


def agg_numerical_column(series):
    return [series.mean()]


def build_anonymized_dataset(dataFrame, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical_variables:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = dataFrame.loc[partition].agg(
            aggregations, squeeze=False)
        sensitive_counts = dataFrame.loc[partition].groupby(
            sensitive_column).agg({sensitive_column: 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column: sensitive_value,
                'count': count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)


def diversity(dataFrame, partition, column):
    return len(dataFrame[column][partition].unique())


def is_l_diverse(dataFrame, partition, sensitive_column, l=2):
    return diversity(dataFrame, partition, sensitive_column) >= l


def t_closeness(dataFrame, partition, column, global_freqs):
    total_count = float(len(partition))
    d_max = None
    group_counts = dataFrame.loc[partition].groupby(column)[
        column].agg('count')
    for value, count in group_counts.to_dict().items():
        p = count/total_count
        d = abs(p-global_freqs[value])
        if d_max is None or d > d_max:
            d_max = d
    return d_max


def is_t_close(dataFrame, partition, sensitive_column, global_freqs, p=0.2):
    if not sensitive_column in categorical_variables:
        raise ValueError("this method only works for categorial values")
    return t_closeness(dataFrame, partition, sensitive_column, global_freqs) <= p


##################################################################################################

# Read the CSV into a Pandas Dataframe
dataFrame = pd.read_csv("./data/adult.all.txt", sep=", ",
                        header=None, names=data_headers, index_col=False, engine='python')

# Assign type to catgorical variables
for header in categorical_variables:
    dataFrame[header] = dataFrame[header].astype('category')

print("Data Preview:")
print(dataFrame.head())

full_spans = get_spans(dataFrame, dataFrame.index)
print(full_spans)


feature_columns = ['age', 'education-num']
sensitive_column = 'income'
finished_partitions = partition_dataset(
    dataFrame, feature_columns, sensitive_column, full_spans, is_k_anonymous)

print(len(finished_partitions))


indexes = build_indexes(dataFrame)
column_x, column_y = feature_columns[:2]
rects = get_partition_rects(
    dataFrame, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

print(rects[:10])

pl.figure(1, figsize=(20, 20))
ax = pl.subplot(1,1,1)
plot_rects(dataFrame, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(dataFrame[column_x], dataFrame[column_y])

dfn = build_anonymized_dataset(
    dataFrame, finished_partitions, feature_columns, sensitive_column)

print(dfn.sort_values(feature_columns+[sensitive_column]))


finished_l_diverse_partitions = partition_dataset(
    dataFrame, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_l_diverse(*args))

print(len(finished_l_diverse_partitions))

column_x, column_y = feature_columns[:2]
l_diverse_rects = get_partition_rects(
    dataFrame, finished_l_diverse_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

pl.figure(2, figsize=(20, 20))
ax = pl.subplot(1,1,1)
plot_rects(dataFrame, ax, l_diverse_rects, column_x,
           column_y, edgecolor='b', facecolor='b')
# plot_rects(dataFrame, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(dataFrame[column_x], dataFrame[column_y])

dfl = build_anonymized_dataset(
    dataFrame, finished_l_diverse_partitions, feature_columns, sensitive_column)

print(dfl.sort_values([column_x, column_y, sensitive_column]))

global_freqs = {}
total_count = float(len(dataFrame))
group_counts = dataFrame.groupby(sensitive_column)[
    sensitive_column].agg('count')
for value, count in group_counts.to_dict().items():
    p = count/total_count
    global_freqs[value] = p

print(global_freqs)

finished_t_close_partitions = partition_dataset(
    dataFrame, feature_columns, sensitive_column, full_spans, lambda *args: is_k_anonymous(*args) and is_t_close(*args, global_freqs))

print(len(finished_t_close_partitions))

dft = build_anonymized_dataset(
    dataFrame, finished_t_close_partitions, feature_columns, sensitive_column)

print(dft.sort_values([column_x, column_y, sensitive_column]))

column_x, column_y = feature_columns[:2]
t_close_rects = get_partition_rects(
    dataFrame, finished_t_close_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

pl.figure(3, figsize=(20, 20))
ax = pl.subplot(1,1,1)
plot_rects(dataFrame, ax, t_close_rects, column_x,
           column_y, edgecolor='b', facecolor='b')
pl.scatter(dataFrame[column_x], dataFrame[column_y])
pl.show()
