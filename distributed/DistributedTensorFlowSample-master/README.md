# DistributedTensorFlowSample

Distributed TensorFlow のサンプルです。


## シングルPC上での実験

grpc_tensorflow_serverがビルドされており、パスも通っている環境で動作します。

- [Distributed TensorFlowを試してみる](http://qiita.com/ashitani/items/2e48729e78a9f77f9790)
- [Distributed TensorFlow でデータ並列を試してみる](http://qiita.com/ashitani/items/dbe76cb9194d60ead9de#_reference-e2760bd7dfd94e6c4dc4)

に説明を書いています。

```
cd standalone
```

シングルPC上でシングルプロセスです。
```
python ./single_cpu.py
```

シングルPC上でモデル並列です。shでサーバを立てて、python からアクセスします。

```
./model_parallel_server.sh
python ./model_parallel.py
```

シングルPC上でデータ並列です。shでサーバを立てて、python からアクセスします。

```
./data_parallel_server.sh
python ./single_cpu.py
```

## GPC上での実験

[こちら](http://qiita.com/ashitani/items/8b52a6b0ca812712a348)を参照してください。

vCPU8個のGCPクラスタが立ち上がってる前提で、

```
export PROJECT_ZONE=YOUR_ZONE
export PROJECT_ID=YOUR_PROJECT
python ./create_tf_servers.py
```

hostsの設定をしたのちに

```
$ kubectl exec -it if /bin/bash
# apt-get install git
# cd home
# git clone https://github.com/ashitani/DistributedTensorFlowSample
# cd DistributedTensorFlowSample/gcp/mnist
# python mnist_distributed.py master
```
