from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from data_utils import Vocabulary


class Model:
    def __init__(self, model_path: str, vocab_path: str):
        self.vocabulary = Vocabulary.from_file(vocab_path)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        saver = tf.train.import_meta_graph('{}.meta'.format(model_path))
        saver.restore(self.session, str(model_path))
        self.input_xs = tf.get_collection('input_xs')[0]
        self.batch_size = tf.get_collection('batch_size')[0]
        self.softmax = tf.get_collection('softmax')[0]
        self.num_steps = 20

    def predict(self, tokens: List[str]) -> np.ndarray:
        tokens = tokens[-self.num_steps + 1:]
        xs = np.zeros([1, self.num_steps])
        n = len(tokens) + 1
        xs[0, :n] = ([self.vocabulary.s_id] +
                     list(map(self.vocabulary.get_id, tokens)))
        feed_dict = {self.input_xs: xs, self.batch_size: 1}
        for v in tf.get_collection('initial_state'):
            # FIXME - well, that's not very general/portable
            state_size = 512 + 2048  # FIXME - epsecially this!
            self.session.run(
                tf.assign(v, tf.zeros([1, state_size]), validate_shape=False))
        return self.session.run(self.softmax[n - 1], feed_dict=feed_dict)

    def predict_top(self, tokens: List[str], top=10) -> List[Tuple[str, float]]:
        probs = self.predict(tokens)
        top_indices = argsort_k_largest(probs, top)
        return [(self.vocabulary.get_token(id_), probs[id_])
                for id_ in top_indices]

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.__exit__(None, None, None)


def argsort_k_largest(x, k):
    if k >= len(x):
        return np.argsort(x)[::-1]
    indices = np.argpartition(x, -k)[-k:]
    values = x[indices]
    return indices[np.argsort(-values)]
