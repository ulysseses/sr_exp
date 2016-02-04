import sys
import time
import yaml
from paper import run_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tower_name', 'tower',
    """If a model is trained with multiple GPU's prefix all Op names with """
    """tower_name to differentiate the operations. Note that this prefix """
    """is removed from the names of all summaries when visualizing a model.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
    "Whether to log device placement.")
tf.app.flags.DEFINE_integer('num_gpus', 4, "How many GPUs to use.")
tf.app.flags.DEFINE_boolean('dev_assign', True, "Do assign tf.devices.")

tf.app.flags.DEFINE_string('command', 'train', "train or infer")
tf.app.flags.DEFINE_string('path_conf', 'paper/test1/a.yaml',
    "Path to configuration file.")


def train(path):
    with open(path, 'r') as f:
        conf = yaml.load(f)
    run_model.train(conf)
    

def infer(path):
    with open(path, 'r') as f:
        conf = yaml.load(f)
    ckpt = tf.train.get_checkpoint_state(conf['path_tmp'])
    if ckpt:
        ckpt = ckpt.model_checkpoint_path
        print('found ckpt: %s' % ckpt)
        time.sleep(2)
    run_model.eval_h5(conf, ckpt)
    time.sleep(2)
    run_model.eval_te(conf, ckpt)


if __name__ == '__main__':
    command = FLAGS.command
    path = FLAGS.path_conf
    if command[:2].lower() == 'tr':
        train(path)
    elif command[:2].lower() == 'in':
        infer(path)
    else:
        raise ValueError('command %s not recognized' % command)
