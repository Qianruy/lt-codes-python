#!/usr/bin/env bash

MAX_JOBS = 16

for i in {1..10}
do 
    echo "Launching process $i"
    python test/run_spatial_coupled_LDPC.py &
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
done

wait

echo "All processes completed."