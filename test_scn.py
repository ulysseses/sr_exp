import yaml
from scn import run_model
import tensorflow as tf

tf.app.flags.DEFINE_string('tower_name', 'tower',
    """If a model is trained with multiple GPU's prefix all Op names with """
    """tower_name to differentiate the operations. Note that this prefix """
    """is removed from the names of all summaries when visualizing a model.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
    "Whether to log device placement.")
tf.app.flags.DEFINE_integer('num_gpus', 1, "How many GPUs to use.")
tf.app.flags.DEFINE_boolean('dev_assign', True, "Do assign tf.devices.")


def main():
    with open('scn/conf.yaml', 'r') as f:
        conf = yaml.load(f)
    run_model.train(conf)


if __name__ == '__main__':
    main()
