#!/bin/zsh

# `repeat` only works in zsh

repeat 3 python longadder.py --trainer.device=mps --data.ndigit=1
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=2
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=3
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=4
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=5
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=6
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=6 --model.model_type=gpt-micro
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=7
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=7 --model.model_type=gpt-micro
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=8
repeat 3 python longadder.py --trainer.device=mps --data.ndigit=8 --model.model_type=gpt-micro
