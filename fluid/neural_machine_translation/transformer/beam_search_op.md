## Beam Search OP in Paddle Fluid

### Beam Search

[wikipedia] Beam search uses breadth-first search to build its search tree. At each level of the tree, it generates all successors of the states at the current level, sorting them in increasing order of heuristic cost. However, it only stores a predetermined number of best states at each level (called the beam width). Only those states are expanded next.

### Beam Search with RNN

Use beam search to get RNN predictions with high probabilities step by step. The following codes show the outline of Beam Search with Vanilla RNN:

```Python
def cell(inputs, states):
    """
    Vanilla RNN cell to transform symbols to logits.
    Typically, inputs are ids of previous step and
    states are hidden states of previous step of RNN.
    """
    word_emb = embedding(inputs)
    next_states = fc([word_emb, states], hidden_size)
    logits = fc([next_states], vocab_size)
    return logits, next_states

pre_array = []
next_array = []

while cur_len < max_len:
    # cell calculations to transform symbols to logits,
    # and then compute the accumulated scores.
    logits, next_states = cell(pre_word_ids, pre_states)
    step_log_probs = log_softmax(logits)
    pre_total_probs += step_log_probs  # [beam_size, 1] + [beam_size, vocab_size]

    # select next words and corresponding parent indices
    next_beam_scores, word_indices = topk(flatten(total_probs))
    next_word_ids = word_indices % vocab_size
    next_beam_ids = word_indices // vocab_size

    # update states and save into array to trace
    pre_states = gather_from(gather_from=next_states, gather_indices==next_beam_ids)
    pre_word_ids = next_word_ids
    pre_total_probs = next_beam_scores
    pre_array += [next_beam_ids]
    next_array += [next_word_ids]

    # If all next words are <eos>, end.
    if is_finish(next_word_ids):
        break
    cur_len += 1

final_seqs = backtrace(pre_array, next_array)
```

### Beam Search by Operators

- [Beam Search step in TensorFlow](https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py#L673)

```Python
# cell calculations to transform inputs to logits
cell_outputs, next_cell_state = self._cell(inputs, cell_state)
# Calculate the total log probs for the new hypotheses
# Final Shape: [batch_size, beam_width, vocab_size]
step_log_probs = nn_ops.log_softmax(logits)
step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs
# Calculate the scores for each beam
scores = _get_scores(
    log_probs=total_probs,  # accumulate score
    sequence_lengths=new_prediction_lengths,  # length penalty
    length_penalty_weight=length_penalty_weight)

# reshape to (batch_size, beam_size * vocab_size)
scores_flat = array_ops.reshape(scores, [batch_size, -1])
next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)
# Pick out the probs, beam_ids, and states according to the chosen predictions
next_word_ids = math_ops.to_int32(
    math_ops.mod(word_indices, vocab_size, name="next_beam_word_ids"))
next_beam_ids = math_ops.to_int32(
    word_indices / vocab_size, name="next_beam_parent_ids")

next_cell_state = nest.map_structure(
    lambda gather_from: _maybe_tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=gather_from,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[batch_size * beam_width, -1]),
    next_cell_state)
next_beam_probs = _tensor_gather_helper(
    gather_indices=word_indices,
    gather_from=total_probs,
    batch_size=batch_size,
    range_size=beam_width * vocab_size,
    gather_shape=[-1],
    name="next_beam_probs")
next_finished = math_ops.logical_or(
    previously_finished,
    math_ops.equal(next_word_ids, end_token),
    name="next_beam_finished")
next_prediction_len = _tensor_gather_helper() +
    math_ops.to_int64(math_ops.logical_not(previously_finished))

next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))
```

- [Beam Search step in Fluid](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_machine_translation.py#L107)

```Python
pre_ids = pd.array_read(array=ids_array, i=counter)
pre_state = pd.array_read(array=state_array, i=counter)
pre_score = pd.array_read(array=scores_array, i=counter)

logits, current_state = cell(pre_ids, pre_state)
topk_scores, topk_indices = pd.topk(pd.softmax(logits), k=beam_size)
# calculate accumulated scores after topk to reduce computation cost
accu_scores = pd.elementwise_add(
    x=pd.log(topk_scores), y=pd.reshape(
        pre_score, shape=[-1]), axis=0)

# select next words and corresponding scores.
# links info between prefixes and selected candidates are organized into lod
selected_ids, selected_scores = pd.beam_search(
    pre_ids,
    pre_score,
    topk_indices,
    accu_scores,
    beam_size,
    end_id=10,
    level=0)

# update states and save into array to trace
current_state = pd.sequence_expand(current_state, pre_score)  # expand(gather) according to lod
pd.increment(x=counter, value=1, in_place=True)
pd.array_write(current_state, array=state_array, i=counter)
pd.array_write(selected_ids, array=ids_array, i=counter)
pd.array_write(selected_scores, array=scores_array, i=counter)

# update the break condition: up to the max length or all candidates of
# source sentences have ended.
length_cond = pd.less_than(x=counter, y=array_len)
finish_cond = pd.logical_not(pd.is_empty(x=selected_ids))
pd.logical_and(x=length_cond, y=finish_cond, out=cond)
```

### How to trace paths in the search tree

- `beam_search_op` in Paddle Fluid
  - Use two-level lod to reserve path links.
    - Level 0 indicates how many prefixes (branchs) for each source sentece.
    Level 1 indicates how the selected candidates belong to theses prefixes.
    - For example, assuming the output's lod is `[[0, 2, 4], [0, 1, 2, 2, 4]]`, then
      - There are two source sentence each with two prefixes(beam width is two).
      - The selected four candidates are successors  to the *1st, 2nd, 4th and 4th* prefixes separately.
      - Thus `sequence_expand_op` expanding by lod can act as `gather_from` to update next states. Here, expand(gather) `[1, 1, 0, 2]` times.
    - Suitable to both LoDTensor(RNNSearch) and Tensor(Transformer); Couple with `sequence_expand_op` and hard to understand.
  - Others:
    - What to do with the end beam
      - The ended branch should not be pruned since it still might be sifted out because of low score.
      - Force the prediction of ended branch to allocate all probability mass to end token.
      - Though the ended branch cann't be pruned , the ended beam whose branches are all finished can be pruned.

- `beam_search_decode_op` in Paddle Fluid
  - Construct hypotheses by walking back along the LoDTensorArray.
  - Organize outputs as LoDTensor containing sequences with different lengths.

### How to parse the outputs

```Python
seq_ids, seq_scores = exe.run(infer_program,
                              feed=data_input,
                              fetch_list=[out_ids, out_scores],
                              return_numpy=False)
########################################################################
# How to parse the results:
#   Suppose the lod of seq_ids is:
#     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
#   then from lod[0]:
#     there are 2 source sentences, beam width is 3.
#   from lod[1]:
#     the first source sentence has 3 hyps; the lengths are 12, 12, 16
#     the second source sentence has 3 hyps; the lengths are 14, 13, 15
########################################################################
hyps = [[] for i in range(len(data))]
scores = [[] for i in range(len(data))]
for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
    start = seq_ids.lod()[0][i]
    end = seq_ids.lod()[0][i + 1]
    for j in range(end - start):  # for each candidate
        sub_start = seq_ids.lod()[1][start + j]
        sub_end = seq_ids.lod()[1][start + j + 1]
        hyps[i].append(" ".join([
            trg_idx2word[idx]
            for idx in post_process_seq(
                np.array(seq_ids)[sub_start:sub_end])
        ]))
        scores[i].append(np.array(seq_scores)[sub_end - 1])
```

### Other than RNN -- Transformer

In Transformer, the inference is still auto regressive and based on beam search. While there is no hidden states and every step needs intermediate outputs of all preceding steps for self-attention in the decoder, must we calculate all of these at each step?

Cache intermediate outputs of all preceding steps rather than previous time step states, thus more memories request.

```Python
def multi_head_attention(queries,
                         keys,
                         values,
                         attn_bias,
                         d_key,
                         d_value,
                         d_model,
                         n_head=1,
                         dropout_rate=0.,
                         pre_softmax_shape=None,
                         post_softmax_shape=None,
                         cache=None):
    def __compute_qkv(queries, keys, values, n_head, d_key, d_value):
        pass

    def __split_heads(x, n_head):
        pass

    def __combine_heads(x):
        pass

    def scaled_dot_product_attention(q, k, v, attn_bias, d_model, dropout_rate):
        pass

    q, k, v = __compute_qkv(queries, keys, values, n_head, d_key, d_value)

    ###########################################################
    if cache is not None:  # use cache and concat time steps
        k = cache["k"] = layers.concat([cache["k"], k], axis=1)
        v = cache["v"] = layers.concat([cache["v"], v], axis=1)
    ###########################################################

    q = __split_heads(q, n_head)
    k = __split_heads(k, n_head)
    v = __split_heads(v, n_head)

    ctx_multiheads = scaled_dot_product_attention(q, k, v, attn_bias, d_model,
                                                  dropout_rate)

    out = __combine_heads(ctx_multiheads)

    # Project back to the model size.
    proj_out = layers.fc(input=out,
                         size=d_model,
                         bias_attr=False,
                         num_flatten_dims=2)
    return proj_out
```

```Python
caches = [{
    "k": layers.fill_constant_batch_size_like(
        input=start_tokens,
        shape=[-1, 0, d_model],
        dtype=enc_output.dtype,
        value=0),
    "v": layers.fill_constant_batch_size_like(
        input=start_tokens,
        shape=[-1, 0, d_model],
        dtype=enc_output.dtype,
        value=0)
} for i in range(n_layer)]

with while_op.block():
    pre_caches = [{
        "k": layers.sequence_expand(
            x=cache["k"], y=pre_scores),
        "v": layers.sequence_expand(
            x=cache["v"], y=pre_scores),
    } for cache in caches]

    # some operators
    op(pre_caches)

    for i in range(n_layer):
        layers.assign(pre_caches[i]["k"], caches[i]["k"])
        layers.assign(pre_caches[i]["v"], caches[i]["v"])
```
