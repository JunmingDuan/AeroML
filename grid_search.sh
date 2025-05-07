#!/bin/bash

nb=5

n=0
for decay in 0.00001
do
  for n_layers in 2 3 4
  do
    for n_hidden in 10 20 30 40
    do
      nohup python main.py --n_in $nb --n_layers $n_layers --n_hidden $n_hidden --weight_decay $decay --normalize > basis${nb}/logs/log_$n &
      n=`expr ${n} + 1`
      echo $n
      sleep 5
    done
  done
done

