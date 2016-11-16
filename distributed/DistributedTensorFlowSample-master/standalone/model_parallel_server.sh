#!/bin/sh
killall grpc_tensorflow_server
grpc_tensorflow_server --cluster_spec='master|localhost:2222,ps|localhost:2223,ps_|localhost:2224' --job_name=master --task_index=0 &
grpc_tensorflow_server --cluster_spec='master|localhost:2222,ps|localhost:2223,ps_|localhost:2224' --job_name=ps --task_index=0 &
grpc_tensorflow_server --cluster_spec='master|localhost:2222,ps|localhost:2223,ps_|localhost:2224' --job_name=ps_ --task_index=0 &