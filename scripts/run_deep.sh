#!/bin/bash

# bash ./scripts/run_deep.sh c10 tqdm 1 "resnet" "" "--dirty_tau --use_mixup_data --prob_leaves_rate 1" ""
# bash ./scripts/run_deep.sh c100 tqdm gumbel no_trainable pretrainModel otherParams date

args=("$@")
for i in "${!args[@]}"; do
  echo "$i: ${args[$i]}"
done

if [[ "${#args[@]}" != "7" ]]; then
  echo "You should use 7 arguments"
  exit 1
fi

if [[ "${args[0]}" == "c100" ]]; then
  echo "USE CIFAR100"
  dataset="CIFAR100"
  dir_name="cifar-100"
  num_leaves=100
  num_nodes=(99 256 512)
  num_jump=(7 12 20)
  batch_size=(4 8)
elif [[ "${args[0]}" == "c10" ]]; then
  echo "USE CIFAR10"
  dataset="CIFAR10"
  dir_name="cifar-10"
  num_leaves=10
  num_nodes=(9 16 32 64)
  num_jump=(4 6 8 10 20)
  batch_size=(128 256)
else
  echo "USE TINYIMAGENET200"
  dataset="TinyImagenet200"
  dir_name=""
  num_leaves=200
  num_nodes=(199 256 512)
  num_jump=(8 16 20 40)
  batch_size=(1 2 3)
fi

save_results=./results_nbdt-${dataset}

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

if [[ "${args[4]}" != "" ]]; then
  echo "PRETRAINED MODEL: ${args[4]}"
  pretrained_model="${args[4]}"
else
  pretrained_model=./pretrained_model
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

      python ./run/main_deep.py --root_data /dataset/$dir_name --dataset $dataset --arch ResNet18 --pretrained --path_resume ./pretrained_model --batch_size $bs --lr $lr --epochs $epochs --num_nodes $nodes --num_leaves $num_leaves --num_jumps $n_jump --num_graphs $num_graphs --results $save_results $use_loss $use_args

      done
    done
  done
done
