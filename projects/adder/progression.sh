#!/bin/bash

for ((i=1; i<=$1; i++))
do
    python longadder.py --trainer.device=mps --data.ndigit=$i --trainer.max_iters=1000000000
done
