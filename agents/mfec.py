"""Model free episodic control with the LSH based dictionary"""
import logging
import os

import numpy as np
import tensorflow as tf

import dnd
from agents.agent import AbstractAgent


class MFECAgent(object):
    """Model free episodic control using the LSH dictionary.
    Only for discrete action spaces (and preferably ones with very few
    actions).

    The agent is implemented as a context manager. This is a slightly atypical
    choice, but it allows it to seamlessly setup and tear down at the
    beginning and end of training runs.
    """

    def __init__(self, num_actions, state_size, logdir, projection_size=64,
                 hash_bits=8, max_neighbours=11, epsilon=0.005,
                 epsilon_steps=1000000, discount=0.99):
        """Sets up initial data structures and a session and initialises.

        Args:
            num_actions (int): the number of possible actions. We keep a
                separate memory for all actions.
            state_size (int): the size of the state vectors that make up the
                observations we will receive.
            logdir (str): path to a directory where we store summaries and
                checkpoints of the memory state
            projection_size (Optional[int]): size of the observations we see,
                probably after either a random projection or some kind of
                learned encoding.
            hash_bits (Optional[int]): how many bits the LSH uses. Directly
                corresponds to the number of buckets we divide the state space
                into (specifically: 2**num_bits). Default is eight, for pretty
                coarse division into 256 buckets.
            max_neighbours (Optional[int]): how many items we store in each
                bucket. This is the number of neighbours in the approximate
                nearest neighbour lookup.
            epsilon (Optional[float]): exploration rate for the final policy.
                This is the chance of picking a move completely at random.
            epsilon_decay_steps (Optional[int]): how many steps we take to
                linearly decay the exploration rate from 1 to `epsilon`.
            discount (Optional[float]): discount rate for the rewards.
        """
        self._logdir = logdir
        self._epsilon = epsilon
        self._gamma = discount
        self._memories = []
        self._store_ops = []
        self._get_ops = []
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._input_pl = tf.placeholder(tf.float32, name='observation',
                                            shape=[state_size])
            self._reward_pl = tf.placeholder(tf.float32, name='reward',
                                             shape=[1])
            with tf.variable_scope('projection'):
                proj_mat = tf.get_variable('projection', trainable=False,
                                           shape=[state_size, projection_size])
                projected_key = tf.matmul(tf.expand_dims(self._input_pl, 0),
                                          proj_mat)
                projected_key = tf.squeeze(projected_key)
            logging.info('getting memory ops')
            for action in range(num_actions):
                with tf.variable_scope('action_{}'.format(action)):
                    # we only have to store a single scalar
                    self._memories.append(dnd.HashDND(
                        hash_bits, max_neighbours, projection_size, [[1]],
                        name='action_{}_memory'.format(action)))
                    self._get_ops.append(self._memories[action].get(
                        projected_key)[0])

                    # only store improvements
                    store_val = tf.maximum(self._get_ops[action],
                                           self._reward_pl)
                    self._store_ops.append(self._memories[action].store(
                        projected_key, [store_val]))

            # set up a decaying epsilon for the policy
            e_step = (1.0 - epsilon) / epsilon_steps
            e_var = tf.get_variable('epsilon', initializer=1.0)
            self._epsilon = tf.maximum(e_var.assign_sub(e_step), epsilon)
            tf.summary.scalar('epsilon', self._epsilon)
            self._act_op = tf.where(tf.random_uniform([1]) > self._epsilon,
                                    tf.argmax(tf.stack(self._get_ops), axis=0),
                                    tf.random_uniform([1], minval=0,
                                                      maxval=num_actions,
                                                      dtype=tf.int64))
            self._act_op = tf.squeeze(self._act_op)

            # get a variable to keep track of how long we've been running
            self._frame_counter = tf.Variable(0, trainable=False,
                                              name='frame_counter')
            self._update_frame_count = self._frame_counter.assign_add(1)

            # get a saver and a summariser
            self._saver = tf.train.Saver(var_list=tf.global_variables())
            self._summary_writer = tf.summary.FileWriter(self._logdir)
            self._summary_writer.add_graph(self._graph)
            self._all_summaries = tf.summary.merge_all()
            self._init_op = tf.group(tf.global_variables_initializer(),
                                     tf.local_variables_initializer())
        logging.info('graph set up complete')

        self._session = None
        # bookkeeping, because we update at the end of an episode
        self._trajectory = []

    def __enter__(self):
        """Set up resources etc..
        In the case of this agent, this means getting a session, initialising
        the variables (either from a file or fresh) and starting up the
        checkpointing/summary services.
        """
        self._session = tf.Session(graph=self._graph)
        # check if we're restoring
        checkpoint = tf.train.latest_checkpoint(self._logdir)
        if checkpoint:
            logging.info('initialising from %s', checkpoint)
            self._saver.restore(self._session, checkpoint)
        else:
            logging.info('initialising from scratch')
            self._session.run(self._init_op)
        logging.info('initialised successfully')
        self._trajectory = []
        return self

    def __exit__(self, exc_value, exc_type, traceback):
        """Release resources. Writes a final checkpoint and closes down the
        session."""
        self._session.close()

    def act(self, observation):
        """Get an action given an observation."""
        observation = np.ravel(observation)
        action, _ = self._session.run([self._act_op, self._update_frame_count],
                                   {self._input_pl: observation})
        self._trajectory.append((observation, action))
        return action

    def reward(self, value):
        """Receive a reward"""
        if len(self._trajectory[-1]) != 2:
            raise ValueError('something is out of order')
        state, action = self._trajectory[-1]
        self._trajectory[-1] = (state, action, value)

    def episode_ended(self):
        """be notified that the current episode has finished"""
        # NOTE: should really be able to batch this up
        # it is also highly likely that this could be implemented substantially
        # more efficiently by actually thinking about it first
        logging.info('storing results of trajectory')
        for i, timestep in enumerate(self._trajectory):
            state, action, reward = timestep
            discounts = np.arange(len(self._trajectory) - i)
            discounts = self._gamma ** discounts
            rewards = np.array([item[-1] for item in self._trajectory[i:]])
            reward_t = np.sum(rewards * discounts)
            # now store it
            self._session.run(self._store_ops[action],
                              {self._input_pl: state,
                               self._reward_pl: [reward_t]})
            # logging.debug('%d: %d, %f, %s', i, action, reward_t, state.shape)

        # write summaries
        logging.info('writing summaries')
        viewed_frames = self._session.run(self._frame_counter)
        self._summary_writer.add_summary(self._session.run(
            self._all_summaries), global_step=viewed_frames)
        total_reward = np.sum([item[-1]
                               for item in self._trajectory])
        logging.info('total reward %f', total_reward)
        self._summary_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(
                tag='average_episode_reward',
                simple_value=total_reward)]),
            viewed_frames)
        # and a checkpoint
        logging.info('saving checkpoint')
        self._saver.save(
            self._session,
            os.path.join(self._logdir, 'model-{}'.format(viewed_frames)))

        # clear the trajectory
        self._trajectory = []
