from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import wmt14

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")


def to_lodtensor(data, place):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    lod_t = core.LoDTensor()
    lod_t.set(flattened_data, place)
    lod_t.set_lod([lod])
    return lod_t, lod[-1]


def train():
    fluid.default_startup_program().random_seed = 111

    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)
    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[30000, 32],
        dtype='float32')

    encoded_proj = fluid.layers.fc(input=src_embedding,
                                   size=32,
                                   bias_attr=False)
    encoded_proj = fluid.layers.Print(encoded_proj, message="encoded_proj", summarize=10)

    decoder_state_proj = fluid.layers.sequence_pool(
        input=encoded_proj, pool_type='last')
    decoder_state_proj = fluid.layers.Print(
        decoder_state_proj, message="decoder_state_proj", summarize=10)

    decoder_state_expand = fluid.layers.sequence_expand(
       x=decoder_state_proj, y=encoded_proj)
    decoder_state_expand = fluid.layers.Print(
        decoder_state_expand, message="decoder_state_expand", summarize=10)

    prediction = fluid.layers.fc(input=decoder_state_expand,
                          size=30000,
                          bias_attr=True,
                          act='softmax')

    prediction = fluid.layers.Print(prediction, message="prediction", summarize=10)
    cost = fluid.layers.cross_entropy(input=prediction, label=src_word_idx)
    avg_cost = fluid.layers.mean(x=cost)

    feeding_list = ["source_sequence"]

    optimizer = fluid.optimizer.Adam(learning_rate=0.01)
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            wmt14.train(30000), buf_size=1000),
        batch_size=16)

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)

    data = [[0], [19], [16], [1]]
    lod = [0, 4]
    lod_t = core.LoDTensor()
    lod_t.set(np.array(data), place)
    lod_t.set_lod([lod])

#    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    for batch_id, data in enumerate(train_batch_generator()):
        src_seq, word_num = to_lodtensor(map(lambda x: x[0], data), place)

        fetch_outs = exe.run(framework.default_main_program(),
                             feed={
                                 feeding_list[0]: lod_t
                             },
                             fetch_list=[avg_cost, prediction.name+"@GRAD", decoder_state_expand.name+"@GRAD"])
        print(fetch_outs)


if __name__ == '__main__':
    args = parser.parse_args()
    train()
