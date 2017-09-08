
# import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import time

import settings as S


def build_cnn(num_filters, filter_sizes, strides, hidden_sizes):
    print("Building CNN")

    input_var = T.tensor4(name='input', dtype=theano.config.floatX)  # float32
    l_in = L.InputLayer(shape=(None,) + S.IMG, input_var=input_var)
    S.print_shape(L.get_output_shape(l_in))

    l_hid = l_in
    for n_filt, filt_size, stride in zip(num_filters, filter_sizes, strides):
        l_hid = L.Conv2DLayer(l_hid,
                              num_filters=n_filt,
                              filter_size=filt_size,
                              stride=stride)
        S.print_shape(L.get_output_shape(l_hid))

    for h_size in hidden_sizes:
        l_hid = L.DenseLayer(l_hid, num_units=h_size)
        S.print_shape(L.get_output_shape(l_hid))

    l_out = L.DenseLayer(l_hid,
                         num_units=S.OUTPUT,
                         nonlinearity=lasagne.nonlinearities.softmax)
    S.print_shape(L.get_output_shape(l_out))
    variables = L.get_all_params(l_out)
    for v in variables:
        print("variable: ", v, " dtype: ", v.dtype)

    return l_out, input_var


def build_inference(output_layer, input_var):
    print("Building inference function")
    test_prediction = L.get_output(output_layer, deterministic=True)
    test_prediction = T.argmax(test_prediction, axis=1)
    f_inference = theano.function(inputs=[input_var], outputs=test_prediction)

    return f_inference


def build_train(output_layer, input_var, target_var):
    print("Building training function")
    train_prediction = L.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=1e-3)
    f_train = theano.function(inputs=[input_var, target_var],
                              outputs=loss,
                              updates=updates)

    return f_train


def main():
    output_layer, input_var = build_cnn(**S.cnn_spec)
    f_inference = build_inference(output_layer, input_var)
    target_var = T.ivector('target')
    f_train = build_train(output_layer, input_var, target_var)

    x, y = S.load_data_numpy()

    print("Warming up and running timing...")
    print("Running inference on batch size: {}".format(S.INF_BATCH))
    for _ in range(5):
        r = f_inference(x[:S.INF_BATCH])

    t_0 = time.time()
    for _ in range(S.REPEAT):
        for i in range(S.INF_BATCHES):
            r = f_inference(x[i * S.INF_BATCH:(i + 1) * S.INF_BATCH])
    t_1 = time.time()
    S.print_time("inference", t_1 - t_0)

    print("Running training on batch size: {}".format(S.TR_BATCH))
    for _ in range(5):
        r = f_train(x[:S.TR_BATCH], y[:S.TR_BATCH])

    t_0 = time.time()
    for _ in range(S.REPEAT):
        for i in range(S.TR_BATCHES):
            r = f_train(x[i * S.TR_BATCH:(i + 1) * S.TR_BATCH],
                        y[i * S.TR_BATCH:(i + 1) * S.TR_BATCH])
    r += 1  # make sure we have it
    t_1 = time.time()
    S.print_time("train", t_1 - t_0)


if __name__ == "__main__":
    main()
