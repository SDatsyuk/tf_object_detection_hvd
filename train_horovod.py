# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Training executable for detection models.

This executable is used to train DetectionModels. There are two ways of
configuring the training job:

1) A single pipeline_pb2.TrainEvalPipelineConfig configuration file
can be specified by --pipeline_config_path.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files can be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being trained, an
input_reader_pb2.InputReader file to specify what training data will be used and
a train_pb2.TrainConfig file to configure training parameters.

Example usage:
    ./train \
        --logtostderr \
        --train_dir=path/to/train_dir \
        --model_config_path=model_config.pbtxt \
        --train_config_path=train_config.pbtxt \
        --input_config_path=train_input_config.pbtxt
"""


import tensorflow as tf
import functools
import json
import os 
import configparser

from google.protobuf import text_format

from object_detection import trainer
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.protos import train_pb2

import horovod.tensorflow as hvd

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string("worker_hosts", None, "Comma sep list of hostname:port pairs")
flags.DEFINE_string("ps_hosts", None, "Comma sep list of hostname:port pairs")
flags.DEFINE_string("task_type", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string('master', '', 'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS

hvd.init()

def change_process_config(pid, path='pid/pid.config'):
  config = configparser.ConfigParser()
  print(FLAGS)
  if os.path.exists(path):
    config.read(path)
    config["TRAIN"]["PID"] = str(pid)
    config["TRAIN"]["train_dir"] = FLAGS.train_dir
    config["TRAIN"]["pipeline_config_path"] = FLAGS.pipeline_config_path

  else:
    config["TRAIN"] = {"PID": pid, "Description": "train", "train_dir": FLAGS.train_dir, 'pipeline_config_path': FLAGS.pipeline_config_path}
  with open(path, 'w') as configfile:
    config.write(configfile)

def get_configs_from_pipeline_file():
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader

  return model_config, train_config, input_config


def get_configs_from_multiple_files():
  """Reads training configuration from multiple config files.

  Reads the training config from the following files:
    model_config: Read from --model_config_path
    train_config: Read from --train_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  train_config = train_pb2.TrainConfig()
  with tf.gfile.GFile(FLAGS.train_config_path, 'r') as f:
    text_format.Merge(f.read(), train_config)

  model_config = model_pb2.DetectionModel()
  with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, train_config, input_config


def main(_):
  assert FLAGS.train_dir, '`train_dir` is missing.'
  if FLAGS.pipeline_config_path:
    model_config, train_config, input_config = get_configs_from_pipeline_file()
  else:
    model_config, train_config, input_config = get_configs_from_multiple_files()

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)

  create_input_dict_fn = functools.partial(
      input_reader_builder.build, input_config)

  # TF_CONFIG={"cluster": {"ps_hosts": "192.168.1.49:2222", "worker_hosts": "192.168.1.49:2223,192.168.1.13:2223"}, "task": {"type": "worker", "index": 0}}
  # os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)
  # input()
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # print(env)
  cluster_data = env.get('cluster', None)
  # cluster_data['ps_hosts'] = cluster_data['ps_hosts'].split(',')
  # cluster_data = {}
  # cluster_data['worker'] = FLAGS.worker_hosts.split(',')
  # cluster_data['ps'] = FLAGS.ps_hosts.split(',')
  # cluster_data['worker'] = cluster_data['worker_hosts']
  # cluster_data['ps'] = cluster_data['ps_hosts']
  if cluster_data:
    cluster_data['worker'] = cluster_data['worker_hosts'].split(',') if cluster_data['worker_hosts'] else None
    cluster_data['ps'] = cluster_data['ps_hosts'].split(',') if cluster_data['ps_hosts'] else None

    cluster = tf.train.ClusterSpec({"ps": cluster_data['ps'], "worker": cluster_data['worker']})

    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)
  # print(task_info.type)

  # Parameters for a single worker.
  ps_tasks = 0
  worker_replicas = 1
  worker_job_name = 'lonely_worker'
  task = 0
  is_chief = True
  master = FLAGS.master

  if cluster_data and 'worker' in cluster_data:
    # Number of total worker replicas include "worker"s and the "master".
    worker_replicas = len(cluster_data['worker']) + 1
    print('worker_replicas: ', worker_replicas)
  if cluster_data and 'ps' in cluster_data:
    ps_tasks = len(cluster_data['ps'])
    print('ps_tasks: ', ps_tasks)

  if worker_replicas > 1 and ps_tasks < 1:
    raise ValueError('At least 1 ps task is needed for distributed training.')

  if worker_replicas >= 1 and ps_tasks > 0:
    # Set up distributed training.
    print('Set up distributed training.')
    # print(type(cluster))
    # input()
    server = tf.train.Server(cluster, protocol='grpc',
                             job_name=FLAGS.task_type,
                             task_index=FLAGS.task_index)
    if FLAGS.task_type == 'ps':
      server.join()
      return

    worker_job_name = '%s/task:%d' % (FLAGS.task_type, FLAGS.task_index)
    print(worker_job_name)
    task = FLAGS.task_index
    is_chief = (FLAGS.task_type == 'master')
    master = server.target
    # master = FLAGS.master
    print(master)
    # input()

  # change_process_config(os.getpid())
  # session_config = None
  session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  session_config.gpu_options.allow_growth = True
  session_config.gpu_options.visible_device_list = str(hvd.local_rank())
  # if FLAGS.task_type == 'master':
  #   session_config = tf.ConfigProto(allow_soft_placement=True, device_filters=['/job:ps', '/job:master'])
  # # Worker should only communicate with itself and ps
  # elif FLAGS.task_type == 'worker':
  #   session_config = tf.ConfigProto(allow_soft_placement=True, device_filters=[
  #       '/job:ps',
  #       '/job:worker/task:%d' % FLAGS.task_index])

  trainer.train(create_input_dict_fn, model_fn, train_config, master, task,
                FLAGS.num_clones, worker_replicas, FLAGS.clone_on_cpu, ps_tasks,
                worker_job_name, is_chief, FLAGS.train_dir, session_config=session_config)


if __name__ == '__main__':
  tf.app.run()
