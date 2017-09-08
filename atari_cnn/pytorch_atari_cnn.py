
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import time

import settings as S


class CNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, strides, hidden_sizes):
        print("Building CNN")
        super().__init__()
        f_in = S.IMG[0]
        x = Variable(torch.rand(S.INF_BATCH, *S.IMG))
        S.print_shape(x.size())
        self.convs = list()
        for i, (num_filt, filt_size, stride) in enumerate(zip(
                num_filters, filter_sizes, strides)):
            conv = nn.Conv2d(in_channels=f_in,
                             out_channels=num_filt,
                             kernel_size=filt_size,
                             stride=stride)
            self.convs.append(conv)
            setattr(self, "conv" + str(i), conv)
            f_in = num_filt
            x = conv(x)
            S.print_shape(x.size())

        self.fcs = list()
        self.conv_out_size = size_in = int(np.prod(x.size()[1:]))
        x = x.view(-1, size_in)
        for i, h_size in enumerate(hidden_sizes + [S.OUTPUT]):
            fc = nn.Linear(size_in, h_size)
            self.fcs.append(fc)
            setattr(self, "fc" + str(i), fc)
            size_in = h_size
            x = fc(x)
            S.print_shape(x.size())

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.view(-1, self.conv_out_size)
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)
        return F.log_softmax(x)


def main():

    cnn = CNN(**S.cnn_spec)
    print(cnn)
    cnn.cuda()
    optimizer = optim.SGD(cnn.parameters(), lr=1e-3)

    x, y = S.load_data_numpy()
    dataset = data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    inference_loader = data.DataLoader(dataset=dataset, batch_size=S.INF_BATCH, shuffle=False)
    train_loader = data.DataLoader(dataset=dataset, batch_size=S.TR_BATCH, shuffle=False)

    print("Warming up and running timing...")
    print("Running inference on batch size: {}".format(S.INF_BATCH))
    for x, _ in inference_loader:
        x = Variable(x.cuda())
        output = cnn(x)
        _, pred = output.data.max(1)

    t_0 = time.time()
    i = 0
    for _ in range(S.REPEAT):
        for x, _ in inference_loader:
            x = Variable(x.cuda())
            output = cnn(x)
            _, pred = output.max(1)
            i += 1
    t_1 = time.time()
    S.print_time("inference", t_1 - t_0, batches=i)

    print("Running training on batch size: {}".format(S.TR_BATCH))
    for x, y in train_loader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        optimizer.zero_grad()
        output = cnn(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

    t_0 = time.time()
    i = 0
    for _ in range(S.REPEAT):
        for x, y in train_loader:
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            output = cnn(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            i += 1
    t_1 = time.time()
    S.print_time("train", t_1 - t_0, batches=i)


if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
    main()

