import os
import time
import multiprocessing as mp
MP_CONTEXT = mp.get_context('spawn')  # NOQA

import numpy as np
import tensorflow as tf
from matplotlib import colors as mcolors

from tensorflow.python.framework import ops
from tensorflow.python.keras.engine.base_layer import Layer

from fen import log as logging
from fen import colors
from fen import constants
from fen import utils
from fen.base_model import BaseModel
from fen.task import Task
from fen.pool import WorkerPool

LOG = logging.getLogger(__name__)

# TODO: REFACTORING


class MyModel(Layer):
    def __init__(self,
                 nb_plays=1,
                 units=1,
                 batch_size=1,
                 weight=1.0,
                 width=1.0,
                 debug=False,
                 activation='tanh',
                 optimizer='adam',
                 timestep=1,
                 input_dim=1,
                 diff_weights=False,
                 network_type=constants.NetworkType.PLAY,
                 learning_rate=0.001,
                 ensemble=1,
                 learnable_mu=False,
                 learnable_sigma=False,
                 parallel_prediction=False,
                 **kwargs):
        super(Layer, self).__init__(**kwargs)
        self._unittest = kwargs.pop('unittest', False)
        if self._unittest is False:
            assert timestep == 1, colors.red('timestep must be 1')

        assert activation in [None, 'tanh', 'relu', 'elu'], colors.red("activation {} not support".format(activation))

        # fix random seed to 123
        # seed = 123
        np.random.seed(ensemble)
        LOG.debug(colors.red("Make sure you are using the right random seed. currently seed is {}".format(ensemble)))

        self.plays = []
        self._nb_plays = nb_plays
        self._units = units
        self._activation = activation
        self._input_dim = input_dim
        self._ensemble = ensemble
        # _weight = 1.0
        _width = 0.1
        # width = 1
        for nb_play in range(nb_plays):
            if diff_weights is True:
                weight = 0.5 / (_width * (1 + nb_play))   # width range from (0.1, ... 0.1 * nb_plays)
                # weight = 0.5 / (_width * (1 + nb_play)) # width range from (0.1, ... 0.1 * nb_plays)
                # weight = nb_play # width range from (0.1, ... 0.1 * nb_plays)
            else:
                weight = 1.0

            # weight = 10 * (nb_play + 1)                  # width range from (0.1, ... 0.1 * nb_plays)
            # weight = 0.02 * (nb_play + 1)                  # width range from (0.1, ... 0.1 * nb_plays)
            weight = 2 * (nb_play + 1)                  # width range from (0.1, ... 0.1 * nb_plays)
            LOG.debug("MyModel {} generates {} with Weight: {}".format(self._ensemble,
                                                                       colors.red("Play #{}".format(nb_play+1)),
                                                                       weight))
            # if debug is True:
            #     weight = 1.0

            play = BaseModel(units=units,
                             batch_size=batch_size,
                             weight=weight,
                             width=width,
                             debug=debug,
                             activation=activation,
                             loss=None,
                             optimizer=None,
                             network_type=network_type,
                             name="play-{}".format(nb_play),
                             timestep=timestep,
                             input_dim=input_dim,
                             unittest=self._unittest,
                             ensemble=self._ensemble)
            assert play._need_compile is False, colors.red("Play inside MyModel mustn't be compiled")
            self.plays.append(play)

        # self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
        # if kwargs.pop("parallel_prediction", False):
        if parallel_prediction:
            self.parallel_prediction = True
            self.pool = WorkerPool(constants.CPU_COUNTS)
            self.pool.start()

        self._learnable_mu = learnable_mu
        self._learnable_sigma = learnable_sigma

    def fit(self,
            inputs,
            outputs,
            epochs=100,
            verbose=0,
            steps_per_epoch=1,
            loss_file_name="./tmp/mymodel_loss_history.csv",
            learning_rate=0.001,
            decay=0.):

        writer = utils.get_tf_summary_writer("./log/mse")

        inputs = ops.convert_to_tensor(inputs, tf.float32)
        __mu__ = (outputs[1:] - outputs[:-1]).mean()
        __sigma__ = (outputs[1:] - outputs[:-1]).std()
        outputs = ops.convert_to_tensor(outputs, tf.float32)

        for play in self.plays:
            if not play.built:
                play.build(inputs)

        _xs = []
        _ys = []
        for play in self.plays:
            _x, _y = play.reshape(inputs, outputs)
            _xs.append(_x)
            _ys.append(_y)

        _xs = [_x]

        params_list = []
        model_outputs = []
        feed_inputs = [utils.get_cache()['play_input_layer'].input]
        feed_targets = []

        for idx, play in enumerate(self.plays):
            # feed_inputs.append(play.input)
            model_outputs.append(play.output)

            shape = tf.keras.backend.int_shape(play.output)
            name = 'play{}_target'.format(idx)
            target = tf.keras.backend.placeholder(
                ndim=len(shape),
                name=name,
                dtype=tf.float32)
            feed_targets.append(target)

            # update_inputs += play.model.get_updates_for(inputs)
            params_list += play.trainable_weights

        if self._nb_plays > 1:
            y_pred = tf.keras.layers.Average()(model_outputs)
        else:
            y_pred = model_outputs[0]

        loss = tf.keras.backend.mean(tf.math.square(y_pred - _y))
        mu = tf.keras.backend.mean(y_pred[:, 1:, :] - y_pred[:, :-1, :])
        sigma = tf.keras.backend.std(y_pred[:, 1:, :] - y_pred[:, :-1, :])

        # decay: decay learning rate to half every 100 steps
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay)
        with tf.name_scope('training'):
            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=params_list,
                                                     loss=loss)
            # updates += update_inputs

            training_inputs = feed_inputs + feed_targets
            train_function = tf.keras.backend.function(training_inputs,
                                                       [loss, mu, sigma],
                                                       updates=updates)

        # _x = [x for _ in range(self._nb_plays)]
        # _y = [y for _ in range(self._nb_plays)]
        self._batch_input_shape = utils.get_cache()['play_input_layer'].input.shape

        ins = _xs + _ys

        self.cost_history = []

        path = "/".join(loss_file_name.split("/")[:-1])
        writer.add_graph(tf.get_default_graph())
        loss_summary = tf.summary.scalar("loss", loss)
        for i in range(epochs):
            for j in range(steps_per_epoch):
                cost, predicted_mu, predicted_sigma = train_function(ins)
            self.cost_history.append([i, cost])
            LOG.debug("Epoch: {}, Loss: {:.7f}, predicted_mu: {:.7f}, predicted_sigma: {:.7f}, truth_mu: {:.7f}, truth_sigma: {:.7f}".format(i,
                                                                                                                                             float(cost),
                                                                                                                                             float(predicted_mu),
                                                                                                                                             float(predicted_sigma),
                                                                                                                                             float(__mu__),
                                                                                                                                             float(__sigma__)))

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1], loss_file_name)

    def predict_parallel(self, inputs, individual=False, states_list=None):
        _inputs = inputs
        if states_list is None:
            states_list = [np.array([0]).reshape(1, 1) for _ in range(self._nb_plays)]
        elif not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]


        ##########################################################################
        #     multiprocessing.Pool
        ##########################################################################
        # pool = MP_CONTEXT .Pool(constants.CPU_COUNTS)
        # args_list = [(play, _inputs) for play in self.plays]
        # outputs = pool.map(parallel_predict, args_list)
        # pool.close()
        # pool.join()
        ##########################################################################
        ##########################################################################
        #     myImplementation.Pool
        ##########################################################################
        start = time.time()
        for i, play in enumerate(self.plays):
            LOG.debug("{}".format(play._name))
            task = Task(play, 'predict', (_inputs, 1, 0, states_list[i]))
            # task = Task(play, 'predict', (_inputs, 1, 0, None))
            self.pool.put(task)

        self.pool.join()
        end = time.time()
        outputs = self.pool.results
        LOG.debug("Cost time {} s".format(end-start))
        ##########################################################################
        #  Serial execution
        ##########################################################################
        # inputs = ops.convert_to_tensor(inputs, tf.float32)
        # x = self.plays[0].reshape(inputs)
        # for play in self.plays:
        #     if not play.built:
        #         play.build(inputs)

        # outputs = []
        # for play in self.plays:
        #     start = time.time()
        #     outputs.append(play.predict(x))
        #     end = time.time()
        #     LOG.debug("play {} cost time {} s".format(play._name, end-start))
        ##########################################################################

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        if individual is True:
            outputs_ = outputs_.reshape(len(self.plays), -1).T
            # sanity checking
            for i in range(len(self.plays)):
                if np.all(outputs_[i, :] == outputs[i]) is False:
                    raise
            return prediction, outputs_
        return prediction

    @property
    def weights(self):
        for play in self.plays:
            LOG.debug("{}, number of layer is: {}".format(play._name, play.number_of_layers))
            LOG.debug("{}, weight: {}".format(play._name, play.weights))

    def _make_batch_input_shape(self, inputs):
        _ = [play._make_batch_input_shape(inputs) for play in self.plays if not play.built]

    def compile(self, inputs, **kwargs):
        mu, sigma = kwargs.pop('mu', None), kwargs.pop('sigma', None)
        outputs = kwargs.pop('outputs', None)
        LOG.debug("Compile with mu: {}, sigma: {}".format(colors.red(mu), colors.red(sigma)))

        _inputs = inputs
        inputs = ops.convert_to_tensor(inputs, tf.float32)
        _ = [play.build(inputs) for play in self.plays if not play.built]

        self.params_list = []
        self.feed_inputs = []
        self.model_outputs = []

        self.feed_mu = tf.constant(mu, name='input-mu', dtype=tf.float32)
        self.feed_sigma= tf.constant(sigma, name='input-sigma', dtype=tf.float32)

        self.mu_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                           name='target-mu',
                                                           dtype=tf.float32)
        self.sigma_placeholder = tf.keras.backend.placeholder(ndim=0,
                                                              name='target-sigma',
                                                              dtype=tf.float32)

        self._batch_input_shape = utils.get_cache()['play_input_layer'].input.shape
        self.feed_inputs = [utils.get_cache()['play_input_layer'].input]
        self.feed_targets = []
        self.update_inputs = []

        for play in self.plays:
            self.model_outputs.append(play.output)
            # TODO: figure out the function of get_updates_for
            self.update_inputs += play.model.get_updates_for(play.input)
            self.params_list += play.trainable_weights

        if self._unittest is True:
            self._x_feed_dict = { self.feed_inputs[0].name : _inputs.reshape(self.batch_input_shape) }

        self._y = [self.feed_mu, self.feed_sigma]

        assert len(self.feed_inputs) == 1, colors.red("ERROR: only need one input layer")
        ##################### Average outputs #############################
        if self._nb_plays > 1:
            self.y_pred = tf.keras.layers.Average()(self.model_outputs)
        else:
            self.y_pred = self.model_outputs[0]

        y_pred = tf.reshape(self.y_pred, shape=(-1,))
        self.diff = tf.math.subtract(y_pred[1:], y_pred[:-1])

        self.diff = tf.concat([tf.reshape(y_pred[0], shape=(1,)), self.diff], axis=0)
        self.curr_mu = tf.keras.backend.mean(self.diff)
        self.curr_sigma = tf.keras.backend.std(self.diff)

        with tf.name_scope('training'):
            if self._unittest is False:
                ###################### Calculate J by hand ###############################
                # 1. gradient for each play, assign to self.J_list_by_hand
                self.J_list_by_hand = [
                    gradient_all_layers(play.operator_layer.output,
                                        play.nonlinear_layer.output,
                                        play.operator_layer.kernel,
                                        play.nonlinear_layer.kernel,
                                        play.linear_layer.kernel,
                                        activation=self._activation) for play in self.plays]
            else:
                ###################### Calculate J by hand ###############################
                # 1. gradient for each play, assign to self.J_list_by_hand
                self.J_list_by_hand = [
                    gradient_all_layers(play.operator_layer.output,
                                        play.nonlinear_layer.output,
                                        play.operator_layer.kernel,
                                        play.nonlinear_layer.kernel,
                                        play.linear_layer.kernel,
                                        activation=self._activation,
                                        debug=True,
                                        inputs=self.feed_inputs[0],
                                        feed_dict=copy.deepcopy(self._x_feed_dict)) for play in self.plays]
                ####################### Calculate J by Tensorflow ###############################
                # 1. gradient for each play, assign to self.J_list_by_tf
                # 2. gradient for summuation of all plays, assign to self.J_by_tf
                model_outputs = [tf.reshape(self.model_outputs[i], shape=self.batch_input_shape) for i in range(self._nb_plays)]
                J_list_by_tf = [tf.keras.backend.gradients(model_output, self.feed_inputs) for model_output in model_outputs]
                self.J_list_by_tf = [tf.reshape(J_list_by_tf[i], shape=(1, -1, 1)) for i in range(self._nb_plays)]

                y_pred = tf.reshape(self.y_pred, shape=self.batch_input_shape)
                J_by_tf = tf.keras.backend.gradients(y_pred, self.feed_inputs)[0]
                self.J_by_tf = tf.reshape(J_by_tf, shape=(1, -1, 1))
            # (T * 1)
            # by hand
            J_by_hand = tf.reduce_mean(tf.concat(self.J_list_by_hand, axis=-1), axis=-1, keepdims=True)
            self.J_by_hand = tf.reshape(J_by_hand, shape=(-1,))

            if self._unittest is True:
                # we don't care about the graidents of loss function, it's calculated by tensorflow.
                return

            normalized_J_by_hand = tf.clip_by_value(tf.abs(self.J_by_hand), clip_value_min=1e-18, clip_value_max=1e18)

            # TODO: support derivation for p0
            # TODO: make loss customize from outside
            # TODO: learn mu/sigma/weights
            # NOTE: current version is fixing mu/sigma and learn weights
            # self._learnable_mu = True
            if self._learnable_mu:
                LOG.debug(colors.red("Using Learnable Mu Version"))
                # import ipdb;ipdb.set_trace()
                # from tensorflow.python.ops import variables as tf_variables
                # from tensorflow.keras.engine import base_layer_utils
                # self.mu = base_layer_utils.make_variable(name='mymodel/mu:0',
                #                                          initializer=tf.keras.initializers.Constant(value=0.0, dtype=tf.float32),
                #                                          trainable=True,
                #                                          dtype=tf.float32)
                # self.mu = tf.VariableV1(0.0, dtype=tf.float32, name='mymodel/mu')
                self._set_dtype_and_policy(dtype=tf.float32)

                self.mu = self.add_weight(
                    "mu",
                    dtype=tf.float32,
                    initializer=tf.keras.initializers.Constant(value=10.0, dtype=tf.float32),
                    trainable=True)
                self.loss_a = tf.keras.backend.square((self.diff - self.mu)/sigma) / 2
                self.params_list.append(self.mu)

            else:
                self.loss_a = tf.keras.backend.square((self.diff - mu)/sigma) / 2
            self.loss_b = - tf.keras.backend.log(normalized_J_by_hand)
            # self.loss_by_hand = tf.keras.backend.mean(self.loss_a + self.loss_b)
            # self.loss_by_hand = tf.keras.backend.sum(self.loss_a + self.loss_b)
            self.loss_by_hand = tf.math.reduce_sum(self.loss_a + self.loss_b)
            # import ipdb; ipdb.set_trace()
            self.loss = self.loss_by_hand

            # 10-14
            self.reg_lambda = 0.001
            self.reg_mu = 0.001
            # 15-19
            # self.reg_lambda = 0.0001
            # self.reg_mu = 0.0001

            regularizers1 = [self.reg_lambda * tf.math.reduce_sum(tf.math.square(play.nonlinear_layer.kernel)) for play in self.plays]
            regularizers2 = [self.reg_mu * tf.math.reduce_sum(tf.math.square(play.linear_layer.kernel)) for play in self.plays]
            for regularizer in regularizers1:
                self.loss += regularizer

            for regularizer in regularizers2:
                self.loss += regularizer

            if outputs is not None:
                mse_loss1 = tf.keras.backend.mean(tf.square(self.y_pred - tf.reshape(outputs, shape=self.y_pred.shape)))
                mse_loss2 = tf.keras.backend.mean(tf.square(self.y_pred + tf.reshape(outputs, shape=self.y_pred.shape)))
            else:
                mse_loss1 = tf.constant(-1.0, dtype=tf.float32)
                mse_loss2 = tf.constant(-1.0, dtype=tf.float32)

            with tf.name_scope(self.optimizer.__class__.__name__):
                updates = self.optimizer.get_updates(params=self.params_list,
                                                     loss=self.loss)
                # TODO: may be useful to uncomment the following code
                # updates += self.update_inputs
            training_inputs = self.feed_inputs + self.feed_targets

            # self.updates = updates
            # self.training_inputs = training_inputs
            if kwargs.get('test_stateful', False):
                outputs = [play.operator_layer.output for play in self.plays]
            elif self._learnable_mu:
                outputs = [self.loss, mse_loss1, mse_loss2, self.diff, self.curr_sigma, self.curr_mu, self.y_pred, tf.keras.backend.mean(self.loss_a), tf.keras.backend.mean(self.loss_b), self.mu] + [play.operator_layer.output for play in self.plays]
            else:
                outputs = [self.loss, mse_loss1, mse_loss2, self.diff, self.curr_sigma, self.curr_mu, self.y_pred, tf.keras.backend.mean(self.loss_a), tf.keras.backend.mean(self.loss_b)] + [play.operator_layer.output for play in self.plays]
            self.train_function = tf.keras.backend.function(training_inputs,
                                                            outputs,
                                                            updates=updates)

    def fit2(self,
             inputs,
             mu,
             sigma,
             outputs=None,
             epochs=100,
             verbose=0,
             steps_per_epoch=1,
             loss_file_name="./tmp/mymodel_loss_history.csv",
             learning_rate=0.001,
             decay=0.,
             preload_weights=False,
             weights_fname=None,
             **kwargs):

        writer = utils.get_tf_summary_writer("./log/mle")

        if outputs is not None:
            # glob ground-truth mu and sigma of outputs
            __mu__ = (outputs[1:] - outputs[:-1]).mean()
            __sigma__ = (outputs[1:] - outputs[:-1]).std()
            outputs = ops.convert_to_tensor(outputs, tf.float32)

        # self.compile(inputs, mu=mu, sigma=sigma, outputs=outputs, **kwargs)
        training_inputs, validate_inputs = inputs[:self._input_dim], inputs[self._input_dim:]
        if outputs is not None:
            training_outputs, validate_outputs = outputs[:self._input_dim], outputs[self._input_dim:]
        else:
            training_outputs, validate_outputs = None, None

        # kwargs['validate_inputs'] = validate_inputs
        # kwargs['validate_outputs'] = validate_outputs
        self.compile(training_inputs, mu=mu, sigma=sigma, outputs=training_outputs, **kwargs)
        # ins = self._x + self._y
        # ins = self._x
        input_dim =  self.batch_input_shape[-1]
        # ins = inputs.reshape(-1, 1, input_dim)
        utils.init_tf_variables()

        writer.add_graph(tf.get_default_graph())
        self.cost_history = []
        cost = np.inf
        patience_list = []
        prev_cost = np.inf

        # load weights pre-trained
        if preload_weights is True and weights_fname is not None:
            # find the best match weights from weights directory
            self.load_weights(weights_fname)

        if kwargs.get('test_stateful', False):
            outputs_list = [[] for _ in range(self._nb_plays)]
            assert epochs == 1, 'only epochs == 1 in unittest'
            for i in range(epochs):
                self.reset_states()
                for j in range(steps_per_epoch):
                    ins = inputs[j*input_dim:(j+1)*input_dim]
                    output = self.train_function([ins.reshape(1, 1, -1)])
                    states_list = [o.reshape(-1)[-1] for o in output]
                    self.reset_states(states_list=states_list)
                    for k, o in enumerate(output):
                        outputs_list[k].append(o.reshape(-1))

            results = []
            for output in outputs_list:
                results.append(np.hstack(output))
            assert len(results) == self._nb_plays
            return results

        if self._learnable_mu:
            logger_string_epoch = "Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}, learned_mu: {:.7f}"
        else:
            logger_string_epoch = "Epoch: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}"
        logger_string_step = "Steps: {}, Loss: {:.7f}, MSE Loss1: {:.7f}, MSE Loss2: {:.7f}, diff.mu: {:.7f}, diff.sigma: {:.7f}, mu: {:.7f}, sigma: {:.7f}, loss_by_hand: {:.7f}, loss_by_tf: {:.7f}, loss_a: {:.7f}, loss_b: {:.7f}"

        for i in range(epochs):
            self.reset_states()
            for j in range(steps_per_epoch):
                ins = inputs[j*input_dim:(j+1)*input_dim]
                if self._learnable_mu:
                    cost, mse_cost1, mse_cost2, diff_res, sigma_res, mu_res, y_pred, loss_a, loss_b, learned_mu, *operator_outputs = self.train_function([ins.reshape(1, 1, -1)])
                else:
                    cost, mse_cost1, mse_cost2, diff_res, sigma_res, mu_res, y_pred, loss_a, loss_b, *operator_outputs = self.train_function([ins.reshape(1, 1, -1)])
                states_list = [o.reshape(-1)[-1] for o in operator_outputs]
                self.reset_states(states_list=states_list)

                if prev_cost <= cost:
                    patience_list.append(cost)
                else:
                    prev_cost = cost
                    patience_list = []
                loss_by_hand, loss_by_tf = 0, 0
            if self._learnable_mu:
                LOG.debug(logger_string_epoch.format(i, float(cost), float(mse_cost1), float(mse_cost2), float(diff_res.mean()), float(diff_res.std()), float(__mu__), float(__sigma__), float(loss_by_hand), float(loss_by_tf), loss_a, loss_b, float(learned_mu)))
            else:
                LOG.debug(logger_string_epoch.format(i, float(cost), float(mse_cost1), float(mse_cost2), float(diff_res.mean()), float(diff_res.std()), float(__mu__), float(__sigma__), float(loss_by_hand), float(loss_by_tf), loss_a, loss_b))

            LOG.debug("================================================================================")
            # save weights every 1000 epochs
            # if i % 1000 == 0 and i != 0:
            #     self.save_weights("{}-epochs-{}.h5".format(weights_fname[:-3], i))

            self.cost_history.append([i, cost, mse_cost1, mse_cost2, loss_a, loss_b])

        cost_history = np.array(self.cost_history)
        tdata.DatasetSaver.save_data(cost_history[:, 0], cost_history[:, 1:], loss_file_name)

    def predict(self, inputs, individual=False):
        if isinstance(inputs, tf.Tensor):
            _inputs = inputs
        else:
            _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)

        _ = [play.build(_inputs) for play in self.plays if play.built is False]
        outputs = [play.predict(inputs) for play in self.plays]

        outputs_ = np.array(outputs)
        prediction = outputs_.mean(axis=0)
        prediction = prediction.reshape(-1)
        if individual is True:
            return prediction, outputs_
        return prediction

    def trend(self, prices, B, mu, sigma,
              start_pos=1000, end_pos=1100,
              delta=0.001, max_iteration=10000):
        # start_pos = 1000
        # # end_pos = 1100
        # end_pos = 1100

        # start_pos = 1000
        # end_pos = 1100
        # end_pos = 1010

        start_pos = 10
        # end_pos = 15
        end_pos = 110

        assert start_pos >= 0, colors.red("start_pos must be larger than 0")
        assert start_pos < end_pos, colors.red("start_pos must be less than end_pos")
        assert len(prices.shape) == 1, colors.red("Prices should be a vector")

        if not hasattr(self, '_batch_input_shape'):
            if hasattr(self.plays[0], '_batch_input_shape'):
                input_dim = self._batch_input_shape.as_list()[-1]
            else:
                raise Exception(colors.red("Not found batch_input_shape"))
        elif isinstance(self._batch_input_shape, (tf.TensorShape, )):
            input_dim = self._batch_input_shape[-1].value
        else:
            raise Exception(colors.red("Unknown **input_dim** error occurs in trend"))

        prices = np.hstack([prices[1500:2000],  prices[0:1000]])

        timestep = prices.shape[0] // input_dim
        shape = (1, timestep, input_dim)
        ################################################################################
        #                  Re-play the noise                                           #
        # original_prediction:                                                         #
        #   - expect the same size of prices                                           #
        # mu: use empirical mean                                                       #
        # sigma: use empirical standard derviation                                     #
        ################################################################################
        original_prediction = self.predict_parallel(prices)
        prices = prices[:original_prediction.shape[-1]]
        real_mu, real_sigma = mu, sigma
        if start_pos > 0:
            mu = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).mean()
            sigma = (original_prediction[1:start_pos] - original_prediction[:start_pos-1]).std()
        mu = 0
        sigma = 110

        LOG.debug(colors.cyan("emprical mean: {}, emprical standard dervation: {}".format(mu, sigma)))
        ################################################################################
        #                Decide the sign of predicted trends                           #
        ################################################################################
        counts = (((prices[1:] - prices[:-1]) >= 0) == ((original_prediction[1:]-original_prediction[:-1]) >= 0))
        sign = None

        # import ipdb; ipdb.set_trace()
        LOG.debug("(counts.sum() / prices.shape[0]) is: {}".format(counts.sum() / prices.shape[0]))
        if (counts.sum() / prices.shape[0]) <= 0.3:
            sign = +1
        elif (counts.sum() / prices.shape[0]) >= 0.7:
            sign = -1
        else:
            raise Exception(colors.red("The neural network doesn't train well"))

        # Enforce prediction to make sense
        original_prediction = original_prediction*sign
        ################################################################################
        #  My Pool
        ################################################################################
        start = time.time()
        for play in self.plays:
            task = Task(play, 'weights', None)
            self.pool.put(task)
        self.pool.join()
        weights_ = self.pool.results
        weights = [[], [], [], [], []]
        for w in weights_:
            weights[0].append(w[0])
            weights[1].append(w[1])
            weights[2].append(w[2])
            weights[3].append(w[3])
            weights[4].append(w[4])

        end = time.time()

        LOG.debug("Time cost during extract weights: {}".format(end-start))
        start = time.time()
        for play in self.plays:
            task = Task(play, 'operator_output', (prices,))
            self.pool.put(task)
        self.pool.join()

        operator_outputs = self.pool.results
        end = time.time()
        LOG.debug("Time cost during extract operator_outputs: {}".format(end-start))

        guess_prices = []
        k = start_pos
        seq = 1
        repeating = 100
        # repeating = 2

        nb_plays = self._nb_plays
        activation = self._activation
        start = time.time()
        pool = MP_CONTEXT.Pool(constants.CPU_COUNTS)
        # pool = MP_CONTEXT.Pool(1)
        args_list = []
        while k + seq - 1 < end_pos:
            prev_gt_price = prices[k-1]
            curr_gt_price = prices[k]
            prev_gt_prediction = original_prediction[k-1]
            curr_gt_prediction = original_prediction[k]
            args = (k,
                    seq,
                    repeating,
                    prev_gt_price,
                    curr_gt_price,
                    prev_gt_prediction,
                    curr_gt_prediction,
                    mu,
                    sigma,
                    nb_plays,
                    activation,
                    sign,
                    operator_outputs,
                    weights,
                    self._ensemble,
                    real_mu,
                    real_sigma)
            args_list.append(args)
            k += 1

        guess_prices = pool.map(wrapper_repeat, args_list)
        pool.close()
        pool.join()
        end = time.time()

        LOG.debug("Time cost for prediction price: {} s".format(end-start))

        LOG.debug("Verifing...")
        # import ipdb; ipdb.set_trace()
        guess_prices = np.array(guess_prices).reshape(-1)

        loss1 =  ((guess_prices - prices[start_pos:end_pos]) ** 2)
        loss2 = np.abs(guess_prices - prices[start_pos:end_pos])
        loss3 = (prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1]) ** 2
        loss4 = np.abs(prices[start_pos:end_pos] - prices[start_pos-1:end_pos-1])

        LOG.debug("hnn-RMSE: {}".format((loss1.sum()/(end_pos-start_pos))**(0.5)))
        LOG.debug("baseline-RMSE: {}".format((loss3.sum()/(end_pos-start_pos))**(0.5)))
        LOG.debug("hnn-L1-ERROR: {}".format((loss2.sum()/(end_pos-start_pos))))
        LOG.debug("baseline-L1-ERROR: {}".format((loss4.sum()/(end_pos-start_pos))))

        return guess_prices

    def visualize_activated_plays(self, inputs, mu=0, sigma=1):
        input_dim = self._batch_input_shape[-1]
        points = inputs.shape[-1]
        timestamp = 1
        shape = (1, timestamp, input_dim)
        _inputs = ops.convert_to_tensor(inputs, dtype=tf.float32)
        _inputs = tf.reshape(_inputs, shape=shape)
        _outputs = [play.operator_layer(_inputs) for play in self.plays]
        outputs = utils.get_session().run(_outputs)
        outputs = [output.reshape(-1) for output in outputs]  # self._nb_plays * intputs.shape(-1)
        outputs = np.array(outputs)
        assert outputs.shape == (self._nb_plays, points)

        import seaborn as sns

        fig, ax = plt.subplots(figsize=(20, 20))
        x_size, y_size = 2, 1
        vmin, vmax = outputs.min(), outputs.max()
        assert x_size * y_size == self._nb_plays

        fig.set_tight_layout(True)
        x = np.linspace(0, x_size+1, x_size+2)
        y = np.linspace(0, y_size+1, y_size+2)
        xv, yv = np.meshgrid(x, y)
        fargs = (outputs, x_size, y_size, vmin, vmax, ax)
        global once
        once = True

        def update(i, *fargs):
            global once
            outputs = fargs[0]
            x_size = fargs[1]
            y_size = fargs[2]
            vmin = fargs[3]
            vmax = fargs[4]
            ax = fargs[5]
            LOG.info("Update animation frame: {}".format(i))
            output = outputs[:, i]
            output = output.reshape(x_size, y_size)
            sns.heatmap(output, linewidth=0.5, vmin=vmin, vmax=vmax, ax=ax, cbar=once)
            once = False

        anim = FuncAnimation(fig, update, frames=np.arange(0, points, 2),
                             fargs=fargs, interval=500)

        fname = './visualize-mu-{}-sigma-{}/heatmap1.gif'.format(mu, sigma)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        anim.save(fname, dpi=40, writer='imagemagick')


        fig, ax = plt.subplots(figsize=(20, 20))
        x_size, y_size = points // 20, 20
        vmin, vmax = outputs.min(), outputs.max()
        assert x_size * y_size == input_dim

        fig.set_tight_layout(True)
        x = np.linspace(0, x_size+1, x_size+2)
        y = np.linspace(0, y_size+1, y_size+2)
        xv, yv = np.meshgrid(x, y)
        fargs = (outputs, x_size, y_size, vmin, vmax, ax)

        once = True
        def update2(i, *fargs):
            global once
            outputs = fargs[0]
            x_size = fargs[1]
            y_size = fargs[2]
            vmin = fargs[3]
            vmax = fargs[4]
            ax = fargs[5]

            LOG.info("Update animation frame: {}".format(i))
            output = outputs[i, :]
            output = output.reshape(x_size, y_size)
            sns.heatmap(output, linewidth=0.5, vmin=vmin, vmax=vmax, ax=ax, cbar=once)
            once = False

        anim = FuncAnimation(fig, update2, frames=np.arange(0, self._nb_plays, 1),
                             fargs=fargs, interval=500)

        fname = './visualize-mu-{}-sigma-{}/heatmap2.gif'.format(mu, sigma)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        anim.save(fname, dpi=40, writer='imagemagick')


    def save_weights(self, fname):
        suffix = fname.split(".")[-1]
        assert suffix == 'h5', "must store in h5 format"

        for play in self.plays:
            path = "{}/{}plays/{}.{}".format(fname[:-3], len(self.plays), play._name, suffix)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'w').close()
            LOG.debug(colors.cyan("Saving {}'s Weights to {}".format(play._name, path)))
            play.save_weights(path)

        LOG.debug(colors.cyan("Writing input shape into disk..."))
        start = time.time()
        # import ipdb; ipdb.set_trace()

        with open("{}/{}plays/input_shape.txt".format(fname[:-3], len(self.plays)), "w") as f:
            f.write(":".join(map(str, self.batch_input_shape)))
        end = time.time()
        LOG.debug("Time cost during writing shape: {} s".format(end-start))

    def load_weights(self, fname, extra={}):
        LOG.debug(colors.cyan("Trying to Load Weights first..."))
        suffix = fname.split(".")[-1]
        dirname = "{}/{}plays".format(fname[:-3], self._nb_plays)
        if extra.get('use_epochs', False) is True or not os.path.isdir(dirname):
            LOG.debug(colors.red("Fail to Load Weights."))
            epochs = []
            base = '/'.join(fname.split('/')[:-1])
            for _dir in os.listdir(base):
                if os.path.isdir('{}/{}'.format(base, _dir)):
                    try:
                        epochs.append(int(_dir.split('-')[-1]))
                    except ValueError:
                        pass

            if not epochs:
                return False
            best_epoch = max(epochs)
            if extra.get('best_epoch', None) is not None:
                best_epoch = extra.get('best_epoch')

            LOG.debug("Loading weights from Epoch: {}".format(epochs))
            dirname = '{}-epochs-{}/{}plays'.format(fname[:-3], best_epoch, self._nb_plays)
            LOG.debug("Loading weights from epochs file: {}".format(dirname))
            if not os.path.isdir(dirname):
                # sanity checking
                raise Exception("Bugs inside *load_wegihts*")

        LOG.debug(colors.red("Found trained Weights. Loading..."))
        with open("{}/input_shape.txt".format(dirname), "r") as f:
            line = f.read()

        shape = list(map(int, line.split(":")))
        if 'shape' in extra:
            shape = extra['shape']

        self._batch_input_shape = tf.TensorShape(shape)

        if getattr(self, 'parallel_prediction'):
            for play in self.plays:
                play._batch_input_shape = tf.TensorShape(shape)
                play._preload_weights = False
                path = "{}/{}.{}".format(dirname, play._name, suffix)
                play._weights_fname = path
        else:
            start = time.time()
            for play in self.plays:
                if not play.built:
                    play._batch_input_shape = tf.TensorShape(shape)
                    play._preload_weights = True
                    play.build()
                path = "{}/{}.{}".format(dirname, play._name, suffix)
                play.load_weights(path)
                LOG.debug(colors.red("Set Weights for {}".format(play._name)))
            end = time.time()
            LOG.debug("Load weights cost: {} s".format(end-start))
        return True

    @property
    def trainable_weights(self):
        weights = []
        for play in self.plays:
            weights.append(play.trainable_weights)

        results = utils.get_session().run(weights)
        for i, result in enumerate(results):
            for j, r in enumerate(result):
                results[i][j] = results[i][j].reshape(-1)

        return results

    def __del__(self):
        LOG.debug("Start to close ProcessPool before deleting object")
        if hasattr(self, 'pool'):
            self.pool.close()

        tf.keras.backend.clear_session()
        utils.get_cache().clear()

    @property
    def states(self):
        # NOTE: doesn't work properly
        return [play.operator_layer.states for play in self.plays]

    def reset_states(self, states_list=None):
        if states_list is None:
            for i, play in enumerate(self.plays):
                play.operator_layer.reset_states()
            return

        if not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]
        for i, play in enumerate(self.plays):
            play.operator_layer.reset_states(states_list[i])

    def reset_states_parallel(self, states_list=None):
        if states_list is None:
            states_list = [np.array([0]).reshape(1, 1) for _ in range(self._nb_plays)]
        elif not isinstance(states_list[0], np.ndarray):
            states_list = [np.array([s]).reshape(1, 1) for s in states_list]
        else:
            states_list = [s.reshape(1, 1) for s in states_list]
        for i, play in enumerate(self.plays):
            task = Task(play, 'reset_states', (states_list[i],))
            self.pool.put(task)
        self.pool.join()

    def get_op_outputs_parallel(self, inputs):
        for play in self.plays:
            task = Task(play, 'operator_output', (inputs,))
            self.pool.put(task)
        self.pool.join()
        return self.pool.results

    @property
    def batch_input_shape(self):
        '''
        return a list
        '''
        return self._batch_input_shape.as_list()

    def _load_sim_dataset(self, i):
        brief_data = np.loadtxt(self._fmt_brief.format(i), delimiter=',')
        truth_data = np.loadtxt(self._fmt_truth.format(i), delimiter=',')
        fake_data = np.loadtxt(self._fmt_fake.format(i), delimiter=',')

        fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = brief_data[0], brief_data[1], brief_data[2], brief_data[3], brief_data[4], brief_data[5]
        fake_price_list, fake_stock_list = fake_data[:, 0], fake_data[:, 1]
        price_list, stock_list = truth_data[:, 0], truth_data[:, 1]
        return fake_price_list, fake_stock_list, price_list, stock_list, fake_B1, fake_B2, fake_B3, _B1, _B2, _B3

    def plot_graphs_together(self, prices, noises, mu, sigma, ensemble_mode=False):
        self._fmt_brief = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-brief.csv'
        self._fmt_truth = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-true-detail.csv'
        self._fmt_fake = '../simulation/training-dataset/mu-0-sigma-110.0-points-2000/{}-fake-detail.csv'
        length = 1000
        assert length <= prices.shape[-1] - 1, "Length must be less than prices.shape-1"
        batch_size = self.batch_input_shape[-1]
        # states_list = None
        prices_tensor = tf.reshape(ops.convert_to_tensor(prices, dtype=tf.float32), shape=(1, 1, -1))

        results = self.predict_parallel(prices)
        self.reset_states_parallel(states_list=None)
        operator_outputs = self.get_op_outputs_parallel(prices)
        self.reset_states_parallel(states_list=None)
        prices = prices[:results.shape[-1]]
        # import ipdb; ipdb.set_trace()
        counts = ((prices[1:]-prices[:-1] >= 0) == (results[1:] - results[:-1] >= 0)).sum()
        sign = None
        if counts / prices.shape[0] >= 0.65:
            sign = -1
        elif counts / prices.shape[0] <= 0.35:
            sign = +1

        LOG.debug("The counts is: {}, percentage is: {}".format(counts, counts/prices.shape[0]))
        if sign is None:
            raise Exception("the neural network doesn't train well, counts is {}".format(counts))

        # determine correct direction of results
        states_list = None

        _result_list = []
        packed_results = []
        bifurcation_list = []

        for i in range(length):
             # fig, (ax1, ax2) = plt.subplots(2, sharex='all')

            fake_price_list, fake_noise_list, price_list, noise_list, fake_B1, fake_B2, fake_B3, _B1, _B2, _B3 = self._load_sim_dataset(i)
            start_price, end_price = price_list[0], price_list[-1]
            if abs(prices[i] - start_price) > 1e-7 or \
              abs(prices[i+1] - end_price) > 1e-7:
                LOG.error("Bugs: prices is out of expectation")

            interpolated_prices = np.linspace(start_price, end_price, batch_size)
            # self.reset_states_parallel(states_list=states_list)
            interpolated_noises = self.predict_parallel(interpolated_prices, states_list=states_list)
            # FOR DEBUG
            _result_list.append(interpolated_noises[-1])
            # result_list.append(interpolated_noises[0])

            fake_start_price, fake_end_price = fake_price_list[0], fake_price_list[-1]
            fake_interpolated_prices = np.linspace(fake_start_price, fake_end_price, batch_size)
            # self.reset_states_parallel(states_list=states_list)
            fake_interpolated_noises = self.predict_parallel(fake_interpolated_prices, states_list=states_list)

            # fake_interpolated_prices = interpolated_prices
            # fake_interpolated_noises = interpolated_noises
            # NOTE: correct here, don't change
            states_list = [o[i] for o in operator_outputs]

            fake_size = fake_price_list.shape[-1]
            if fake_size >= 50:
                if fake_size // 50 <= 1:
                    fake_size = 100
                fake_price_list_ = fake_price_list[::fake_size // 50]
                fake_price_list = np.hstack([fake_price_list_, fake_price_list[-1]])
                fake_noise_list_ = fake_noise_list[::fake_size // 50]
                fake_noise_list = np.hstack([fake_noise_list_, fake_noise_list[-1]])

            size = price_list.shape[-1]
            if size >= 50:
                if size // 50 <= 1:
                   size = 100
                price_list_ = price_list[::size // 50]
                price_list = np.hstack([price_list_, price_list[-1]])
                noise_list_ = noise_list[::size // 50]
                noise_list = np.hstack([noise_list_, noise_list[-1]])

            # fake_price_list = price_list
            # fake_noise_list = noise_list

            # price_list = fake_price_list
            # noise_list = fake_noise_list


            # import ipdb; ipdb.set_trace()
            fake_size = fake_price_list.shape[-1]
            fake_step = batch_size//fake_size
            if fake_step <= 0:
                fake_step = 1

            fake_interpolated_prices_ = fake_interpolated_prices[::fake_step]
            fake_interpolated_noises_ = fake_interpolated_noises[::fake_step]
            fake_interpolated_prices = np.hstack([fake_interpolated_prices_, fake_interpolated_prices[-1]])
            fake_interpolated_noises = np.hstack([fake_interpolated_noises_, fake_interpolated_noises[-1]])

            size = price_list.shape[-1]
            step = batch_size // size
            if step <= 0:
                step = 1
            interpolated_prices_ = interpolated_prices[::step]
            interpolated_noises_ = interpolated_noises[::step]
            interpolated_prices = np.hstack([interpolated_prices_, interpolated_prices[-1]])
            interpolated_noises = np.hstack([interpolated_noises_, interpolated_noises[-1]])

            # fake_interpolated_prices = interpolated_prices
            # fake_interpolated_noises = interpolated_noises

            # interpolated_prices = fake_interpolated_prices
            # interpolated_noises = fake_interpolated_noises


            interpolated_noises = sign * interpolated_noises
            fake_interpolated_noises = sign * fake_interpolated_noises

            if ensemble_mode is True:
                packed_results.append([fake_price_list, fake_noise_list, price_list, noise_list,
                                       fake_interpolated_prices, fake_interpolated_noises, interpolated_prices, interpolated_noises,
                                       fake_B1, fake_B2, fake_B3, _B1, _B2, _B3])
                continue

            fig, ax1 = plt.subplots(1, figsize=(10, 10))

            is_bifurcation_1, is_correct_direction_of_noise_1 = self._plot_sim(ax1, fake_price_list, fake_noise_list,
                                                                               price_list, noise_list, fake_B1,
                                                                               fake_B2, fake_B3, _B1, _B2, _B3)
            is_bifurcation_2, is_correct_direction_of_noise_2 = self._plot_interpolated(ax1, fake_interpolated_prices, fake_interpolated_noises,
                                                                                        interpolated_prices, interpolated_noises, fake_B1,
                                                                                        fake_B2, fake_B3, _B1, _B2, _B3)
            # is_bifurcation_1, is_correct_direction_of_noise_1 = self._detect_bifurcation(price_list, noise_list)
            # is_bifurcation_2, is_correct_direction_of_noise_2 = self._detect_bifurcation(interpolated_prices, interpolated_noises)
            bifurcation_list.append([is_bifurcation_1, is_bifurcation_2, is_correct_direction_of_noise_1, is_correct_direction_of_noise_2])
            # import ipdb; ipdb.set_trace()
            if mu is None and sigma is None:
                fname = './frames/{}.png'.format(i)
            else:
                fname = './frames-nb_plays-{}-units-{}-batch_size-{}-mu-{}-sigma-{}/ensemble-{}/{}.png'.format(
                    self._nb_plays, self._units, self._input_dim,
                    mu, sigma,
                    self._ensemble,
                    i)

            LOG.debug("plot {}".format(fname))
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            fig.savefig(fname, dpi=100)

        fname = './frames-nb_plays-{}-units-{}-batch_size-{}-mu-{}-sigma-{}/ensemble-{}/bifurcation.csv'.format(
            self._nb_plays, self._units, self._input_dim,
            mu, sigma,
            self._ensemble)
        # import ipdb; ipdb.set_trace()
        bifurcation_list = np.array(bifurcation_list).astype(int)

        np.savetxt(fname, bifurcation_list, fmt="%s", delimiter=',', header='#ground-truth-bifurcation,#neural-network-bifurcation,#ground-truth-direction,#neural-network-direction')  # NOQA

        return packed_results

        # result_list.append(interpolated_noises[-1])
        # result_list = np.array(result_list)
        # import ipdb; ipdb.set_trace()
        # if np.allclose(results, result_list) is True:
        #     print("Correct")
        # else:
        #     import ipdb; ipdb.set_trace()
        #     print("Hello world")
    def _detect_bifurcation(self, price_list, noise_list):
        # Detect bifurcation and predict correct noise ?
        if price_list[-1] - price_list[0] > 0:
            # price rises, find maximum value of noise
            h1 = abs(np.max(noise_list))
        else:
            # price decreases, find minimum value of noise
            h1 = abs(np.min(noise_list))

        h2 = np.max(noise_list) - np.min(noise_list)
        ratio = h1/h2

        flag = not ((price_list[-1] > price_list[0]) ^ (noise_list[-1] < noise_list[0]))

        return (ratio >= 0.1), flag

    def _plot_sim(self, ax,
                  fake_price_list, fake_noise_list,
                  price_list, noise_list,
                  fake_B1, fake_B2, fake_B3,
                  _B1, _B2, _B3, color='blue', plot_target_line=True):

        fake_l = 10 if len(fake_price_list) == 1 else len(fake_price_list)
        l = 10 if len(price_list) == 1 else len(price_list)  # NOQA
        fake_B1, fake_B2, fake_B3 = np.array([fake_B1]*fake_l), np.array([fake_B2]*fake_l), np.array([fake_B3]*fake_l)
        _B1, _B2, _B3 = np.array([_B1]*l), np.array([_B2]*l), np.array([_B3]*l)

        if plot_target_line is True:
            fake_B2 = fake_B2 - fake_B1
            fake_B3 = fake_B3 - fake_B1
            fake_noise_list = fake_noise_list - fake_B1
            fake_B1 = fake_B1 - fake_B1

            _B2 = _B2 - _B1
            _B3 = _B3 - _B1
            noise_list = noise_list - _B1
            _B1 = _B1 - _B1
            ax.plot(fake_price_list, fake_B1, 'r', fake_price_list, fake_B2, 'c--', fake_price_list, fake_B3, 'k--')
            ax.plot(price_list, _B1, 'r', price_list, _B2, 'c', price_list, _B3, 'k-')

        else:
            fake_noise_list = fake_noise_list - fake_B1
            fake_B1 = fake_B1 - fake_B1
            noise_list = noise_list - _B1
            _B1 = _B1 - _B1
            pass

        # import ipdb; ipdb.set_trace()
        ax.plot(fake_price_list, fake_noise_list, color=color, marker='s', markersize=3, linestyle='--')
        ax.plot(price_list, noise_list, color=color, marker='.', markersize=6, linestyle='-')
        ax.set_xlabel("Prices")
        ax.set_ylabel("#Noise")


        is_bifurcation, is_correct_direction_of_noise = self._detect_bifurcation(price_list, noise_list)
        if is_bifurcation:
            ax.text(1.1*price_list.mean(), 0.9*noise_list.mean(), "bifurcation", color=color)
        else:
            ax.text(0.9*price_list.mean(), 0.9*noise_list.mean(), "non-bifurcation", color=color)
        if is_correct_direction_of_noise:
            ax.text(price_list.mean(), noise_list.mean(), 'True', color=color)
        else:
            ax.text(price_list.mean(), noise_list.mean(), 'False', color=color)

        return is_bifurcation, is_correct_direction_of_noise

            # Detect bifurcation and predict correct noise ?
        # if price_list[-1] - price_list[0] > 0:
        #     # price rises, find maximum value of noise
        #     h1 = abs(np.max(noise_list))
        # else:
        #     h1 = abs(np.min(noise_list))
        # h2 = np.max(noise_list) - np.min(noise_list)
        # ratio = h1/h2

        # if ratio >= 0.1:
        #     ax.text(1.1*price_list.mean(), 0.9*noise_list.mean(), "bifurcation", color=color)
        #             # horizontalalignment='right',
        #             # verticalalignment='bottom',
        #             # transform=ax.transAxes)
        # else:
        #     ax.text(0.9*price_list.mean(), 0.9*noise_list.mean(), "non-bifurcation", color=color)
        #     # ax.text(0.75, 0.8, "non-bifurcation", color=color,
        #     #         horizontalalignment='right',
        #     #         verticalalignment='bottom',
        #     #         transform=ax.transAxes)
        # # import ipdb; ipdb.set_trace()
        # flag = not (price_list[-1] > price_list[0]) ^ (noise_list[-1] < noise_list[0])
        # if flag:
        #     ax.text(price_list.mean(), noise_list.mean(), 'True', color=color)
        # else:
        #     ax.text(price_list.mean(), noise_list.mean(), 'False', color=color)

    def _plot_interpolated(self, ax,
                           fake_interpolated_prices,
                           fake_interpolated_noises,
                           interpolated_prices,
                           interpolated_noises,
                           fake_B1, fake_B2, fake_B3,
                           _B1, _B2, _B3,
                           color=mcolors.CSS4_COLORS['orange']):

        return self._plot_sim(ax,
                              fake_interpolated_prices, fake_interpolated_noises,
                              interpolated_prices, interpolated_noises,
                              fake_interpolated_noises[0], fake_B2, fake_B3,
                              interpolated_noises[0], _B2, _B3, color, plot_target_line=False)
