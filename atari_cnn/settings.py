
import numpy as np


IMG = (4, 84, 84)
OUTPUT = 10

INF_BATCH = 16
TR_BATCH = 128

TR_BATCHES = 50
DATA = TR_BATCH * TR_BATCHES
INF_BATCHES = DATA // INF_BATCH
REPEAT = 10

small_cnn_spec = dict(
    num_filters=[16, 32],
    filter_sizes=[8, 4],
    strides=[4, 2],
    hidden_sizes=[512],
)

large_cnn_spec = dict(
    num_filters=[32, 64, 64],
    filter_sizes=[8, 4, 3],
    strides=[4, 2, 1],
    hidden_sizes=[512],
)

# Choose one:
CNN = "small"
# CNN = "large"

if CNN == "small":
    cnn_spec = small_cnn_spec
elif CNN == "large":
    cnn_spec = large_cnn_spec
else:
    raise ValueError("Unrecognized CNN requested: ", CNN)


def load_data_numpy():
    print("Generating synthetic data")
    x = np.random.randn(DATA, *IMG).astype("float32")
    y = np.random.randint(low=0, high=OUTPUT - 1, size=DATA).astype("int32")
    return x, y


def print_time(inference_or_train, time, batches=None):
    if inference_or_train == "inference":
        if batches is None:
            batches = INF_BATCHES * REPEAT
        print("Ran inference on {} batches (total data: {}) in {:.3f} s".format(
            batches, INF_BATCHES * INF_BATCH * REPEAT, time))
    elif inference_or_train == "train":
        if batches is None:
            batches = TR_BATCHES * REPEAT
        print("Ran training on {} batches (total data: {}) in {:.3f} s".format(
            batches, DATA * REPEAT, time))
    else:
        raise ValueError("Unrecognized inference_or_train: ", inference_or_train)


layer_idx = 0


def print_shape(shape):
    global layer_idx
    print("layer ", layer_idx, " output shape: ", shape)
    layer_idx += 1
