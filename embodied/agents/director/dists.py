import embodied
import ruamel.yaml as yaml
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
from tensorflow_probability import distributions as tfd
import numpy as np

from . import behaviors
from . import nets
from . import tfagent
from . import tfutils

class DynDist(tfutils.Module):

    def __init__(
            self, config, act='elu', layers = 4, units = 128, out_dim=1, input_type='feat', normalize_input=False):

        self.config = config
        self._layers = layers
        self._num_layers = layers
        self._units = units
        self._act = get_act(act)
        self.out_dim = out_dim
        
        self._input_type = input_type
        self._normalize_input = normalize_input
        self._dd_opt = tfutils.Optimizer('dyn_dist', **config.dd_opt)

    def __call__(self, gc_obs, no_softmax=False):
        if self._normalize_input:
            _inp, _goal = tf.split(gc_obs, 2, axis = -1)
            _inp = _inp/(tf.norm(_inp)+1e-8)
            _goal = _goal/(tf.norm(_goal)+1e-8)
            x = tf.concat([_inp, _goal], axis=-1)
        else:
            x = gc_obs

        for index in range(self._layers):
            x = self.get(f'fc{index}', nets.Dense, self._units, self._act)(x)

        out = tf.squeeze(self.get(f'hout', nets.Dense, self.out_dim)(x))
        if self.out_dim <= 1 or no_softmax:
            return out
        else:
            return tf.nn.softmax(out)
        
    def train(self, data):
        pass

    def update(self, traj, world_model):
        metrics = {}
        with tf.GradientTape() as df_tape:
            if self.config.gc_input == 'embed':
                _inp = world_model.heads['embed'](traj).mode()
            elif self.config.gc_input == 'state':
                _inp = world_model.heads['decoder'](traj)[self.state_key].mode()
                _inp = tf.cast(self.obs2goal(_inp), self.dtype)
            dd_loss, mets5 = self.loss(_inp)
        metrics.update(self._dd_opt(df_tape, dd_loss, self.dynamical_distance))

    def loss(self, _data, corr_factor = None):
        metrics = {}
        seq_len, bs = _data.shape[:2]
        # pred = tf.cast(self.dynamical_distance(tf.concat([_data, _data], axis=-1)), tf.float32)
        # _label = 1.0
        # loss = tf.reduce_mean((_label-pred)**2)
        # return loss, metrics

        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
            goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)
            pred = tf.cast(self.__call__(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)

            if self.config.dd_loss == 'regression':
                _label = distance
                if self.config.dd_norm_reg_label and self.config.dd_distance == 'steps_to_go':
                    _label = _label/self.dd_seq_len
                loss += tf.reduce_mean((_label-pred)**2)
            else:
                _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
                loss += self.dd_loss_fn(_label, pred)
            return loss

        #positives
        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
        loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])
        metrics['dd_pos_loss'] = loss

        #negatives
        corr_factor = corr_factor if corr_factor != None else self.config.dataset.length
        if self.config.dd_neg_sampling_factor>0:
            num_negs = int(self.config.dd_neg_sampling_factor*self.config.dd_num_positives)
            neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
            neg_loss = _helper(neg_cur_idxs, neg_goal_idxs, tf.ones(num_negs)*seq_len)
            loss += neg_loss
            metrics['dd_neg_loss'] = neg_loss

        return loss, metrics


def get_act(name):
    if name == 'none':
        return tf.identity
    if name == 'mish':
        return lambda x: x * tf.math.tanh(tf.nn.softplus(x))
    elif hasattr(tf.nn, name):
        return getattr(tf.nn, name)
    elif hasattr(tf, name):
        return getattr(tf, name)
    else:
        raise NotImplementedError(name)

def get_future_goal_idxs(seq_len, bs):

        cur_idx_list = []
        goal_idx_list = []
        #generate indices grid
        for cur_idx in range(seq_len):
            for goal_idx in range(cur_idx, seq_len):
                cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
                goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))

        return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
        cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        for i in range(num_negs):
            goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
        return cur_idxs, goal_idxs