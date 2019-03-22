# collect training data from p-pi instead of p-data
# 1. train p-pi(a|o)   √
# 2. run p-pi(a|o) to get dataset o1, o2.. --> D-pi   √ 
# 3. label D-pi with action a_t ==> policy fn, expert_policy_file
#    action = policy_fn(obs[None,:])   √ 
# 4. aggregate D <- D unit D-pi   √

import tensorflow as tf
import numpy as np 
import pickle
import gym
import time
from sklearn.utils import gen_batches
import load_policy

def network(intput_data, output_data, batch_size = 512):
    # 1. build network to train:  
    input_obs = tf.placeholder(tf.float32, shape = [None, data_obs.shape[-1]])
    output_action_true = tf.placeholder(tf.float32, shape = [None, data_acts.shape[-1]])

    hidden1 = tf.contrib.layers.fully_connected(inputs = input_obs, num_outputs = 200)
    hidden2 = tf.contrib.layers.fully_connected(inputs = hidden1, num_outputs = 128)
    hidden3 = tf.contrib.layers.fully_connected(inputs = hidden2, num_outputs = 64)
    output = tf.contrib.layers.fully_connected(inputs = hidden3, num_outputs = data_acts.shape[-1], activation_fn = None)

    loss = tf.losses.mean_squared_error(labels = output_action_true, predictions = output)
    # tf.summary.scalar('loss', loss)
    train = tf.train.AdamOptimizer().minimize(loss)
    return input_obs, output, output_action_true, loss, train 

def dagger(data_obs, data_acts, num_rollouts = 20):
    # prepare policy_fn
    policy_fn = load_policy.load_policy('experts/Humanoid-v2.pkl')
    index = np.arange(data_obs.shape[0]) 

    input_obs, output, output_action_true, loss, train = network(data_obs, data_acts)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        begin = time.time()
        # training phase
        print("-----training phase1-----")
        print('obs:', data_obs.shape)
        print('acts:', data_acts.shape)
        
        for i in range(epoch):
            all_loss = []
            np.random.shuffle(index)
            for batch in gen_batches(data_obs.shape[0], batch_size = 512):
                res_loss, _ = sess.run([loss, train], feed_dict = {input_obs: data_obs[index[batch]], output_action_true: data_acts[index[batch]]})
                all_loss.append(res_loss)
            all_loss = np.mean(all_loss)
            print('epoch {} -> loss {}'.format(i, res_loss))
            
        env = gym.make(envname)
        max_steps = max_timestamps or env.spec.timestep_limit

        observations = []
        actions = []
        for i in range(num_rollouts): 
            obs = env.reset()
            action = sess.run(output, feed_dict = {input_obs: obs[None, :]}) # using network 
            # 2. to get dateset o1,o2.. D-pi
            obs, r, done, _ = env.step(action)
            observations.append(obs)
            # 3. label a_t with dataset
            action_expert = policy_fn(obs[None,:])
            action_expert = np.array(action_expert).flatten()
            actions.append(action_expert)
        print(len(observations))
        print(len(actions))
        # 4. dagger D with D-pi, unit
        data_obs = np.concatenate((data_obs, np.array(observations)))
        data_acts = np.concatenate((data_acts, np.array(actions)))

        # train the network with new data altogether
        print("-----training phase2-----")
        print('obs:', data_obs.shape)
        print('acts:', data_acts.shape)

        for i in range(epoch):
            all_loss = []
            np.random.shuffle(index)
            for batch in gen_batches(data_obs.shape[0], batch_size = 512):
                res_loss, _ = sess.run([loss, train], feed_dict = {input_obs: data_obs[index[batch]], output_action_true: data_acts[index[batch]]})
                all_loss.append(res_loss)
            all_loss = np.mean(all_loss)
            print('epoch {} -> loss {}'.format(i, res_loss))

        print("-----testing phase-----")
        # testing phase (will see the performance of dagger)
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts): 
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                ### Question: the network hasn't been updated? ###
                action = sess.run(output, feed_dict={input_obs: obs[None,:]}) 
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                # if args.render:
                #     env.render()  
                if steps % 100 == 0: 
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns:', returns)
        print('mean return:', np.mean(returns))
        print('std of return:', np.std(returns))  

def load_data():
    # 1.1 load data
    with open('expert_data/Humanoid-v2.pkl', 'rb') as f:
        data = pickle.loads(f.read())
    data_obs = data['observations']
    data_acts = data['actions']
    data_obs = data_obs.reshape(data_obs.shape[0], data_obs.shape[1]) # [2000, 376]
    data_acts = data_acts.reshape(data_acts.shape[0], data_acts.shape[2]) # [2000, 17]
    return data_obs, data_acts

if __name__ == '__main__':
    # some argument:
    envname = 'Humanoid-v2'      
    max_timestamps = None
    render = True
    epoch = 30
    data_obs, data_acts = load_data()
    print(data_acts.shape)
    print(data_obs.shape)

    dagger(data_obs, data_acts)

# Result:
# epoch: 30
# num_rollouts: 20
# mean return: 10398.51344787046
# std of return: 54.72523069729358
