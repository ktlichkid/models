export GLOG_v=1
export GLOG_logtostderr=1
export CUDA_VISIBLE_DEVICE='0'
python machine_translation.py --pass_num=10000 --device CPU
