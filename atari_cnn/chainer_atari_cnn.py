
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import time

import settings as S


class CNN(chainer.Chain):

    def __init__(self, num_filters, filter_sizes, strides, hidden_sizes):
        super().__init__()
        print("Building CNN")
        with self.init_scope():
            self.convs = list()
            f_in = S.IMG[0]
            x = chainer.Variable(
                np.random.randn(S.INF_BATCH, *S.IMG).astype("float32"))
            S.print_shape(x.shape)
            for i, (num_filt, filt_size, stride) in enumerate(zip(
                    num_filters, filter_sizes, strides)):
                conv = L.Convolution2D(in_channels=f_in,
                                       out_channels=num_filt,
                                       ksize=filt_size,
                                       stride=stride)
                x = conv(x)
                S.print_shape(x.shape)
                self.convs.append(conv)
                setattr(self, "conv" + str(i), conv)
                f_in = num_filt
            self.fcs = list()
            size_in = None
            for i, h_size in enumerate(hidden_sizes + [S.OUTPUT]):
                fc = L.Linear(size_in, h_size)
                x = fc(x)
                S.print_shape(x.shape)
                self.fcs.append(fc)
                setattr(self, "fc" + str(i), fc)
                size_in = h_size

    def __call__(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        if chainer.config.train:
            return self.fcs[-1](x)
        return F.softmax(self.fcs[-1](x))


def main():
    cnn = CNN(**S.cnn_spec)
    model = L.Classifier(cnn)
    model.to_gpu()
    optimizer = chainer.optimizers.SGD(lr=1e-3)
    optimizer.setup(model)
    x, y = S.load_data_numpy()

    print("warming up and running timing...")
    print("Running inference on batch size: {}".format(S.INF_BATCH))
    for _ in range(5):
        x_v = chainer.cuda.to_gpu(x[:S.INF_BATCH])
        pred = cnn(x_v)
        pred = pred.data.max(axis=1)

    t_0 = time.time()
    for _ in range(S.REPEAT):
        for i in range(S.INF_BATCHES):
            x_v = chainer.cuda.to_gpu(x[i * S.INF_BATCH:(i + 1) * S.INF_BATCH])
            pred = cnn(x_v)
            pred = pred.data.max(axis=1)
    t_1 = time.time()
    S.print_time("inference", t_1 - t_0)

    print("Running training on batch size: {}".format(S.TR_BATCH))
    for _ in range(5):
        x_v = chainer.Variable(chainer.cuda.to_gpu(x[:S.TR_BATCH]))
        y_v = chainer.Variable(chainer.cuda.to_gpu(y[:S.TR_BATCH]))
        optimizer.update(model, x_v, y_v)

    t_0 = time.time()
    for _ in range(S.REPEAT):
        for i in range(S.TR_BATCHES):
            x_v = chainer.Variable(chainer.cuda.to_gpu(x[i * S.TR_BATCH:(i + 1) * S.TR_BATCH]))
            y_v = chainer.Variable(chainer.cuda.to_gpu(y[i * S.TR_BATCH:(i + 1) * S.TR_BATCH]))
            optimizer.update(model, x_v, y_v)
    t_1 = time.time()
    S.print_time("train", t_1 - t_0)


if __name__ == "__main__":
    main()
