#!/usr/bin/env bash

# time python run_sweep.py 0 5 5 no & python run_sweep.py 1 5 5 no & python run_sweep.py 2 5 5 no & python run_sweep.py 3 5 5 no & python run_sweep.py 4 5 5 no & python run_sweep.py 5 5 5 no
# time python run_sweep.py 1 local 2 2 no
# python run_sweep.py 2 2 2 no &
# python run_sweep.py 3 2 2 no &
# python run_sweep.py 4 2 2 no &
# python run_sweep.py 5 2 2 no &

python inference.py 0 & python inference.py 1 & python inference.py 2 & python inference.py 3 & python inference.py 4 & python inference.py 5