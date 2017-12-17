#!/bin/bash

python main.py --cuda --plot --shuffle --initial google --pause_value 1 --epochs 50 --lim 1000 --lr 0.005 --batch_size 100
python main.py --cuda --plot --shuffle --initial google --pause_value 1 --epochs 50 --lim 1000 --lr 0.001 --batch_size 200

python main.py --cuda --plot --shuffle --initial google --pause_value 1 --epochs 50 --lim 10000 --lr 0.001 --batch_size 100
python main.py --cuda --plot --shuffle --initial google --pause_value 1 --epochs 50 --lim 10000 --lr 0.001 --batch_size 200
python main.py --cuda --plot --shuffle --initial google --pause_value 1 --epochs 50 --lim 10000 --lr 0.001 --batch_size 400
