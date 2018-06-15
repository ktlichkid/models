from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")


def train():
    fluid.default_startup_program().random_seed = 111

    test_data_1 = fluid.layers.data(
        name='test_data_1', shape=[4], dtype='int64', lod_level=1)

    test_data_2 = fluid.layers.data(
        name='test_data_2', shape=[1], dtype='int64', lod_level=1)

    data_1_expanded = fluid.layers.sequence_expand(
        x=test_data_1, y=test_data_2)

    data_1_fc = fluid.layers.fc(
        input=data_1_expanded, size=32, act='softmax')

    cost = fluid.layers.cross_entropy(input=data_1_fc, label=test_data_2)

    avg_cost = fluid.layers.mean(x=cost)

    feeding_list = ["test_data_1", "test_data_2"]

    optimizer = fluid.optimizer.Adam(learning_rate=0.1)
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)

    data_1 = [[1, 3, 17, 2]]
    data_2 = [[0], [19], [16], [1]]
    lod_1 = [[0, 1]]
    lod_2 = [[0, 4]]
    tensor_1 = core.LoDTensor()
    tensor_1.set(np.array(data_1), place)
    tensor_2 = core.LoDTensor()
    tensor_2.set(np.array(data_2), place)
    tensor_1.set_lod(lod_1)
    tensor_2.set_lod(lod_2)

    exe = Executor(place)
    exe.run(framework.default_startup_program())

    print(framework.default_main_program())

    for i in range(0, 2):
        fetch_outs = exe.run(framework.default_main_program(),
                             feed={
                                 feeding_list[0]: tensor_1,
                                 feeding_list[1]: tensor_2
                             },
                             fetch_list=[avg_cost])
        for out in fetch_outs:
            print(out)


if __name__ == '__main__':
    args = parser.parse_args()
    train()
