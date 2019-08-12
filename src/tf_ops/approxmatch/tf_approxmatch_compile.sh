#!/usr/bin/env bash

set -e
set -x

cd "$(dirname "$0")"

TF_PATH=$1

/usr/local/cuda/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $TF_PATH/include -I /usr/local/cuda/include -I $TF_PATH/include/external/nsync/public -lcudart -L /usr/local/cuda/lib64/ -L/$TF_PATH -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
