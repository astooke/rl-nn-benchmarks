
import numpy as np
import tensorflow as tf
import time

import settings as S


def build_cnn(num_filters, filter_sizes, strides, hidden_sizes):
    print("Building CNN")

    input_var = tf.placeholder(dtype=tf.float32,
                               shape=(None,) + S.IMG,
                               name='input')
    S.print_shape(input_var.shape)

    l_hid = input_var
    for n_filt, filt_size, stride in zip(num_filters, filter_sizes, strides):
        l_hid = tf.layers.conv2d(inputs=l_hid,
                                 filters=n_filt,
                                 kernel_size=filt_size,
                                 strides=stride,
                                 activation=tf.nn.relu,
                                 data_format='channels_first')
        S.print_shape(l_hid.shape)

    l_hid = tf.reshape(l_hid, [-1, int(np.prod(l_hid.shape[1:]))])

    for h_size in hidden_sizes:
        l_hid = tf.layers.dense(inputs=l_hid,
                                units=h_size,
                                activation=tf.nn.relu)
        S.print_shape(l_hid.shape)

    logits = tf.layers.dense(inputs=l_hid,
                             units=S.OUTPUT,
                             activation=None)
    S.print_shape(logits.shape)
    variables = tf.global_variables()
    for v in variables:
        print("variable: ", v)
    # print(variables)  # check for float32 dtype

    return logits, input_var


def build_inference(logits, input_var):
    print("Building inference")
    prediction = tf.argmax(tf.nn.softmax(logits), 1)
    return prediction


def build_train(logits, input_var, target_var, var_list=None):
    print("Building training")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_var,
        logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy)
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    train_step = opt.minimize(cross_entropy)

    return train_step


def main():
    logits, input_var = build_cnn(**S.cnn_spec)
    f_inference = build_inference(logits, input_var)
    target_var = tf.placeholder(dtype=tf.int32, shape=(None,), name="target")
    f_train = build_train(logits, input_var, target_var)

    x, y = S.load_data_numpy()

    print("Warming up and running timing...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Running inference on batch size: {}".format(S.INF_BATCH))
        for _ in range(5):
            r = f_inference.eval(feed_dict={input_var: x[:S.INF_BATCH]})

        t_0 = time.time()
        for _ in range(S.REPEAT):
            for i in range(S.INF_BATCHES):
                r = f_inference.eval(
                    feed_dict={input_var: x[i * S.INF_BATCH:(i + 1) * S.INF_BATCH]})
        t_1 = time.time()
        S.print_time("inference", t_1 - t_0)

        print("Running training on batch size: {}".format(S.TR_BATCH))
        for _ in range(5):
            f_train.run(feed_dict={input_var: x[:S.TR_BATCH],
                                      target_var: y[:S.TR_BATCH]})

        t_0 = time.time()
        for _ in range(S.REPEAT):
            for i in range(S.TR_BATCHES):
                f_train.run(feed_dict={input_var: x[i * S.TR_BATCH:(i + 1) * S.TR_BATCH],
                                           target_var: y[i * S.TR_BATCH:(i + 1) * S.TR_BATCH]})
        t_1 = time.time()
        S.print_time("train", t_1 - t_0)

        # print("getting trace")
        # run_metadata = tf.RunMetadata()
        # _ = sess.run(f_train,
        #            feed_dict={input_var: x[:TR_BATCH], target_var: y[:TR_BATCH]},
        #            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #            run_metadata=run_metadata)

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())
        # print("trace written")


if __name__ == "__main__":
    main()
