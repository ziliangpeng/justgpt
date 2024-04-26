#!/bin/bash

for i in {1..4}
do
    python longadder.py --trainer.device=mps --data.ndigit=$i
done
