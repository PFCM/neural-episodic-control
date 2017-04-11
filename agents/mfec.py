"""Model free episodic control with the LSH based dictionary"""
import tensorflow as tf
import gym

import dnd


class MFECAgent(object):
    """Model free episodic control using the LSH dictionary.
    Only for discrete action spaces (and preferably ones with very few
    actions).

    The agent is implemented as a context manager. This is a slightly atypical
    choice, but it allows it to seamlessly setup and tear down at the
    beginning and end of training runs.
    """

    def __init__(self, num_actions, logdir, projection_size=64, hash_bits=8,
                 max_neighbours=5):
        """Sets up initial data structures and a session and initialises.

        Args:
            num_actions (int): the number of possible actions. We keep a
                separate memory for all actions.
            logdir (str): path to a directory where we store summaries and
                checkpoints of the memory state.
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
        """
        self._memories = {}
        for action in range(num_actions):
            # we only have to store a single scalar
            self._memories[action] = dnd.HashDND(
                hash_bits, max_neighbours, projection_size, [[1]],
                name='action_{}_memory'.format(action))

        # get a variable to keep track of how long we've been running
        self._frame_counter = tf.Variable(0, trainable=False,
                                          name='frame_counter')

    def __enter__(self):
        """Set up resources etc..
        In the case of this agent, this means getting a session, initialising
        the variables (either from a file or fresh) and starting up the
        checkpointing/summary services.
        """
        pass

    def __exit__(self):
        """Release resources. Writes a final checkpoint and closes down the
        session."""
        pass
