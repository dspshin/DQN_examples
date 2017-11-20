'''
This code is based on
https://github.com/hunkim/DeepRL-Agents

CF https://github.com/golbin/TensorFlow-Tutorials
https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py
'''

import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn import dqn

from my_gym import gym

#env = gym.make('CartPole-v2')
env = gym

# Constants defining our neural network
input_size = env.input_size
output_size = len(env.action_space)

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predic(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # get target from target DQN (Q')
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack( [x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def ddqn_replay_train(mainDQN, targetDQN, train_batch):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def simple_replay_train(mainDQN, train_batch):
    '''
    Simple DQN implementation
    :param mainDQN: main DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)

    # Get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # Double DQN: y = r + gamma * targetDQN(s')[a] where
            # a = argmax(mainDQN(s'))
            Q[0, action] = reward + dis * np.max(mainDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, isTest=False):
    # See our trained network in action in test env
    state = env.reset(isTest)
    start = state[2]
    reward_sum = 0
    while True:
        #env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("{} based profit : {}".format("Test" if isTest else "Train", reward_sum))
            break

    end = state[2]
    # test 기간 초에 샀다가 마지막에 팔 경우의 점수 (기준점수)...
    print('default profit :', end-start)

def main():
    max_episodes = 200
    # store the previous observations in replay memory
    replay_buffer = deque()

    env.load("005930")

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()

        #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(max_episodes):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            reward_sum = 0

            while not done:
                if np.random.rand(1) < e:
                    action = random.sample( env.action_space, 1 )[0]
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                # if done:  # ends
                #     reward = -100

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                      replay_buffer.popleft()

                state = next_state
                step_count += 1
                # if step_count > 10000:   # Good enough. Let's move on
                #     break

            #print("Episode: {} rewards: {}".format(episode, reward_sum))

            if episode % 10 == 1: # train every 10 episode
                # Get a random batch of experiences
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 100)
                    loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch)
                    #loss, _ = simple_replay_train(mainDQN, minibatch)

                #print("Loss: ", loss)
                #print("Episode: {}, Loss: {}".format(episode, loss))
                print("Episode:{}, rewards:{}, loss:{}".format(episode, reward_sum, loss))

                # copy q_net -> target_net
                sess.run(copy_ops)

        print("Loss:", loss)

        #for i in range(10):
        bot_play(mainDQN, False) # training result
        bot_play(mainDQN, True) # test result


if __name__ == "__main__":
    main()





