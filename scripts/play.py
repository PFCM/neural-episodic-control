"""Play some games with a particular agent."""
import argparse
import logging

from skimage.transform import resize
from skimage.color import rgb2gray

import gym

import agents


def get_preprocessed_frame(observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]


def get_agent(args, env):
    """Get agent by name"""
    if args.agent == 'random':
        agent = agents.RandomAgent(env.action_space.n)
    elif args.agent == 'mfec':
        agent = agents.MFECAgent(env.action_space.n,
                                 84*84, args.logdir,
                                 hash_bits=args.hash_bits,
                                 projection_size=args.projection_size,
                                 epsilon_steps=args.epsilon_steps)
    else:
        raise ValueError('unknown agent: {}'.format(args.agent))
    return agent


def get_env(args):
    """Get the environment to play"""
    env = gym.make(args.env)
    return env


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--agent', '-a', help='which agent',
                        choices=['mfec', 'random'], default='random')
    parser.add_argument('--hash_bits', type=int, default=8,
                        help='bits for LSH')
    parser.add_argument('--projection_size', '-p', type=int, default=64,
                        help='size of projection before hashing')
    parser.add_argument('--max_neighbours', type=int, default=5,
                        help='how many neighbours in approx nearest neighbour')
    parser.add_argument('--logdir', default='./results',
                        help='where to save stuff')
    parser.add_argument('--epsilon_steps', default=1000000, type=int,
                        help='how fast to decay the exploration')

    parser.add_argument('--env', '-e', default='Pong-v0',
                        help='Name of the gym environment to use.')
    parser.add_argument('--max_episodes', type=int, default=10000,
                        help='How many games to play.')
    parser.add_argument('--render', action='store_true',
                        help='whether to draw the game.')
    return parser.parse_args()


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    args = _parse_args()
    env = get_env(args)

    with get_agent(args, env) as agent:
        for episode in range(args.max_episodes):
            print('Episode: {}'.format(episode))
            observation = env.reset()
            done = False
            while not done:
                if args.render:
                    env.render()
                observation = get_preprocessed_frame(observation)
                action = agent.act(observation)
                observation, reward, done, _ = env.step(action)
                agent.reward(reward)
            # must be done
            agent.episode_ended()
            print('  final reward: {}'.format(reward))


if __name__ == '__main__':
    main()
