export GLOG_v=1
export GLOG_logtostderr=1
export CUDA_VISIBLE_DEVICE='2'
python attention_seq2seq.py --infer_only