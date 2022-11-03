#!/bin/bash

# bash ./scripts/run_shallow.sh mnist t 1 "representation" "" "--dirty_tau --prob_leaves_rate 1" ""
# bash ./scripts/run_shallow.sh data tqdm gumbel no_trainable pretrainModel otherParams date

args=("$@")
for i in "${!args[@]}"; do
  echo "$i: ${args[$i]}"
done

if [[ "${#args[@]}" != "7" ]]; then
  echo "You should use 7 arguments"
  exit 1
fi


if [[ "${args[0]}" == "mnist" ]]; then
  echo "USE MNIST"
  dir_name="mnist"
  num_leaves=10
  num_nodes=(9 16 32 64 128)
  num_jump=(4 6 8 10 20)
  batch_size=(64 128)
  save_results=./results_nbdt-MNIST
  dim=784
elif [[ "${args[0]}" == "letter" ]]; then
  echo "USE LETTER"
  dir_name="letter"
  num_leaves=26
  num_nodes=(25 32 64 128 256)
  num_jump=(5 10 20 30 40 50)
  batch_size=(64 128)
  save_results=./results_nbdt-LETTER
  dim=16
else
  echo "USE CONNECT4"
  dir_name="connect4"
  num_leaves=3
  num_nodes=(2 8 16 32)
  num_jump=(2 5 10)
  batch_size=(64 128)
  save_results=./results_nbdt-CONNECT4
  dim=126
fi

use_args=""

if [[ "${args[1]}" != "" ]]; then
  echo "USE TQDM"
  if [[ "$use_args" == "" ]]; then
    use_args="--use_tqdm"
  else
    use_args="$use_args --use_tqdm"
  fi
fi

if [[ "${args[2]}" != "" ]]; then
  echo "USE Gumbel-Softmax (val: ${args[2]})"
  if [[ "$use_args" == "" ]]; then
    use_args="--tau ${args[2]}"
  else
    use_args="$use_args --tau ${args[2]}"
  fi
  save_results="${save_results}_GumbelSoftmax-${args[2]}"
fi

if [[ "${args[3]}" != "" ]]; then
  echo "NO TRAINABLE WEIGHTS ${args[3]}"
  if [[ "$use_args" == "" ]]; then
    use_args="--no_trainable ${args[3]}"
  else
    use_args="$use_args --no_trainable ${args[3]}"
  fi
  save_results="${save_results}_frozen-${args[3]// /-}"
fi

pretrained_model="."
if [[ "${args[4]}" != "" ]]; then
  echo "PRETRAINED MODEL: ${args[4]}"
  pretrained_model="${args[4]}"
  if [[ "$use_args" == "" ]]; then
    use_args="--use_representation ${pretrained_model}/mnist_cnn.pt --layers_nodes 50"
  else
    use_args="$use_args --use_representation ${pretrained_model}/mnist_cnn.pt --layers_nodes 50"
  fi
else
  use_args="$use_args --layers_nodes $dim"
fi

if [[ "${args[5]}" != "" ]]; then
  echo "ADDITIONAL PARAMS ${args[5]}"
  if [[ "$use_args" == "" ]]; then
    use_args="${args[5]}"
  else
    use_args="$use_args ${args[5]}"
  fi
fi

if [[ "${args[6]}" != "" ]]; then
  save_results="${save_results}_${args[6]}"
else
  save_results="${save_results}_$(date +'%Y-%m-%d')"
fi

echo "ARGS: $use_args"
echo "SAVE RESULTS: $save_results"
echo "=================================================================="

##############################################################################
#             grid of selected parameters for dataset examples
##############################################################################

if [[ ! -d "$save_results" ]]; then
  mkdir "$save_results"
fi

use_loss=""
epochs=250
num_graphs=5

export PYTHONPATH=.

for lr in 1e-3; do
  for bs in ${batch_size[*]}; do
    for nodes in ${num_nodes[*]}; do
      for n_jump in ${num_jump[*]}; do

      python ./run/main_shallow.py --root_data ./dataset --data_type $dir_name --batch_size $bs --lr $lr --epochs $epochs --num_nodes $nodes --num_leaves $num_leaves --num_jumps $n_jump --num_graphs $num_graphs --results $save_results $use_loss $use_args

      done
    done
  done
done
