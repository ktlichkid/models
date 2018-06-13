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
    "--batch_size",
    type=int,
    default=16,
    help="The sequence number of a mini-batch data. (default: %(default)d)")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="Learning rate used to train the model. (default: %(default)f)")
parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")


def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim):

    src_word_idx = fluid.layers.data(
        name='source_sequence', shape=[1], dtype='int64', lod_level=1)

    src_embedding = fluid.layers.embedding(
        input=src_word_idx,
        size=[source_dict_dim, embedding_dim],
        dtype='float32')

    trg_word_idx = fluid.layers.data(
        name='target_sequence', shape=[1], dtype='int64', lod_level=1)

    trg_embedding = fluid.layers.embedding(
        input=trg_word_idx,
        size=[target_dict_dim, embedding_dim],
        dtype='float32')

    label = fluid.layers.data(
        name='label_sequence', shape=[1], dtype='int64', lod_level=1)

    encoded_proj = fluid.layers.fc(input=src_embedding,
                                   size=encoder_size,
                                   bias_attr=False)

    decoder_boot = fluid.layers.sequence_pool(
        input=encoded_proj, pool_type='last')

    rnn = fluid.layers.DynamicRNN()

    with rnn.block():
        current_word = rnn.step_input(trg_embedding)
        encoder_proj = rnn.static_input(encoded_proj)
        hidden_mem = rnn.memory(init=decoder_boot)

        decoder_state_proj = hidden_mem
        decoder_state_proj = fluid.layers.Print(
            decoder_state_proj, message="decoder_state_proj", summarize=10)
        decoder_state_expand = fluid.layers.sequence_expand(
           x=decoder_state_proj, y=encoder_proj)
        decoder_state_expand = fluid.layers.Print(
            decoder_state_expand, message="decoder_state_expand", summarize=10)
        concated = fluid.layers.concat(
          input=[encoder_proj, decoder_state_expand], axis=1)
        concated = fluid.layers.Print(
           concated, message="concated", summarize=10)
        context = fluid.layers.sequence_pool(input=concated, pool_type='sum')

        decoder_inputs = fluid.layers.concat(
            input=[context, current_word], axis=1)

        output_gate = fluid.layers.fc(input=[hidden_mem, decoder_inputs], size=decoder_size, bias_attr=True)
 #       cell_tilde = fluid.layers.fc(input=[hidden_mem, decoder_inputs], size=decoder_size, bias_attr=True)

        h = output_gate

#        h = fluid.layers.elementwise_mul(
#            x=output_gate, y=cell_tilde)

        rnn.update_memory(hidden_mem, h)
        out = fluid.layers.fc(input=h,
                              size=target_dict_dim,
                              bias_attr=True,
                              act='softmax')
        rnn.output(out)

    prediction = rnn()

    prediction = fluid.layers.Print(prediction, message="prediction", summarize=10)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    feeding_list = ["source_sequence", "target_sequence", "label_sequence"]

    return avg_cost, feeding_list


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


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ndarray = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ndarray.ravel()[i] = lod_tensor.get_float_element(i)
    return ndarray


def train():
    fluid.default_startup_program().random_seed = 111

    avg_cost, feeding_list = seq_to_seq_net(32, 32, 32, 30000, 30000)

    optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
    optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    train_batch_generator = paddle.batch(
        paddle.reader.shuffle(
            wmt14.train(30000), buf_size=1000),
        batch_size=args.batch_size)

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    for batch_id, data in enumerate(train_batch_generator()):
        src_seq, word_num = to_lodtensor(map(lambda x: x[0], data), place)
        trg_seq, word_num = to_lodtensor(map(lambda x: x[1], data), place)
        lbl_seq, _ = to_lodtensor(map(lambda x: x[2], data), place)

        fetch_outs = exe.run(framework.default_main_program(),
                             feed={
                                 feeding_list[0]: src_seq,
                                 feeding_list[1]: trg_seq,
                                 feeding_list[2]: lbl_seq
                             },
                             fetch_list=[avg_cost])


if __name__ == '__main__':
    args = parser.parse_args()
    train()
