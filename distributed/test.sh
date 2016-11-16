#!/usr/bin/env bash

COUNT_PS_HOSTS=2
PS_HOSTS=localhost:2222,localhost:2223
COUNT_WORKER_HOSTS=2
WORKER_HOSTS=localhost:2777,localhost:2778

for INDEX in `seq 0 $(expr $COUNT_PS_HOSTS - 1)`
do
    python2 client.py \
            --ps_hosts=$PS_HOSTS \
            --worker_hosts=$WORKER_HOSTS \
            --job_name=ps \
            --task_index=$INDEX |& tee ps.${INDEX}.log &
done

for INDEX in `seq 0 $(expr $COUNT_WORKER_HOSTS - 1)`
do
    python2 client.py \
            --ps_hosts=$PS_HOSTS \
            --worker_hosts=$WORKER_HOSTS \
            --job_name=worker \
            --task_index=$INDEX |& tee worker.${INDEX}.log &
done