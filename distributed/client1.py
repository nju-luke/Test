import tensorflow as tf
import numpy as np

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "192.168.0.103:13242",
                    "List of hostname:port for ps jobs."
                    "This string should be the same on every host!!")
tf.app.flags.DEFINE_string("worker_hosts", "192.168.0.103:13243", #, 172.16.122.128:13241",
                    "List of hostname:port for worker jobs."
                    "This string should be the same on every host!!")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "ps", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start1 a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            x = tf.placeholder("float", [10, 10], name="x")
            y = tf.placeholder("float", [10, 1], name="y")
            initial_w = np.zeros((10, 1))
            w = tf.Variable(initial_w, name="w", dtype="float32")
            loss = tf.pow(tf.add(y,-tf.matmul(x,w)),2,name="loss")
            global_step = tf.Variable(0)

            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            while not sv.should_stop() and step < 1000:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                _, step = sess.run([loss, global_step],
                {
                  x: np.random.rand(10,10),
                  y: np.random.rand(10).reshape(-1,1)
                })
                print("job_name: %s; task_index: %s; step: %d" % (FLAGS.job_name,FLAGS.task_index,step))

        # Ask for all the services to stop.
        sv.stop()


if __name__ == "__main__":
    tf.app.run()