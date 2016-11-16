#!/bin/sh
killall grpc_tensorflow_server
grpc_tensorflow_server --cluster_spec='master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name=master --task_index=0 &
grpc_tensorflow_server --cluster_spec='master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name=worker --task_index=0 &
grpc_tensorflow_server --cluster_spec='master|localhost:2222,worker|localhost:2223,worker_|localhost:2224' --job_name=worker_ --task_index=0 &
