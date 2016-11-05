import os.path
import time

import tensorflow as tf

from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval


flags = tf.flags
flags.DEFINE_string('logdir', None, 'Logging directory.')
flags.DEFINE_string('datadir', None, 'Data directory.')
flags.DEFINE_string('vocab', None, 'Vocab file.')
flags.DEFINE_string(
    'mode', 'train',
    'Whether to run "train" or "eval_(train|valid|test)" model.')
flags.DEFINE_string('hpconfig', '', 'Overrides default hyper-parameters.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs used.')
flags.DEFINE_integer('eval_steps', 70, 'Number of eval steps.')

FLAGS = flags.FLAGS


def main(_):
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps.num_gpus = FLAGS.num_gpus

    vocab = Vocabulary.from_file(FLAGS.vocab)
    hps.vocab_size = vocab.num_tokens

    if FLAGS.mode == 'train':
        if FLAGS.logdir is None:
            FLAGS.logdir = os.path.join('/tmp', 'lm-run-{}'.format(int(time.time())))
            print('logdir: {}'.format(FLAGS.logdir))
        hps.batch_size = 256
        dataset = Dataset(vocab, FLAGS.datadir + '/train-*')
        run_train(dataset, hps, FLAGS.logdir + '/train', ps_device='/gpu:0')
    elif FLAGS.mode.startswith('eval_'):
        assert FLAGS.logdir is not None
        if FLAGS.mode == 'eval_train':
            data_dir = FLAGS.datadir + '/train-*'
        elif FLAGS.mode == 'eval_valid':
            data_dir = FLAGS.datadir + '/valid*'
        elif FLAGS.mode == 'eval_test':
            data_dir = FLAGS.datadir + '/test*'
        else:
            raise ValueError('Unknown mode {}'.format(FLAGS.mode))
        dataset = Dataset(vocab, data_dir, deterministic=True)
        run_eval(dataset, hps, FLAGS.logdir, FLAGS.mode, FLAGS.eval_steps)


if __name__ == '__main__':
    tf.app.run()
