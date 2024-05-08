#!/bin/bash

for ((i=1; i<=$1; i++))
do
    python longadder.py --trainer.device=mps --data.ndigit=$i --trainer.max_iters=1000000000 --model.n_layer=3 --model.n_head=2 --model.n_embd=32
done
