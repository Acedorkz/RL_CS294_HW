import pickle
import tensorflow as tf
import numpy as np
import time
import gym

def get_batch(obs, acts, batch_size = 512): # num of batches for a epoch
    index = np.arange(obs.shape[0])
    np.random.shuffle(index)
    batch_index = []
    for i in index:
        batch_index.append(i)
        if len(batch_index) == batch_size:
            yield batch_index
            batch_index = []
    if len(batch_index) != 0:
        yield batch_index

def main():
    # some argument:
    envname = 'Humanoid-v2'      
    max_timestamps = None
    num_rollouts = 20
    render = True
    #
    
    # load expert data:
    with open('expert_data/Humanoid-v2.pkl', 'rb') as f:
        data = pickle.loads(f.read())

    obs_data = data['observations']
    acts_data = data['actions']
    obs_data = obs_data.reshape(obs_data.shape[0], obs_data.shape[1]) # [2000, 376]
    acts_data = acts_data.reshape(acts_data.shape[0], acts_data.shape[2]) # [2000, 17]

    # build tf graph:
    input_obs = tf.placeholder(tf.float32, shape = [None, obs_data.shape[-1]])
    output_action_true = tf.placeholder(tf.float32, shape = [None, acts_data.shape[-1]])

    hidden1 = tf.contrib.layers.fully_connected(inputs = input_obs, num_outputs = 200)
    hidden2 = tf.contrib.layers.fully_connected(inputs = hidden1, num_outputs = 128)
    hidden3 = tf.contrib.layers.fully_connected(inputs = hidden2, num_outputs = 64)
    output = tf.contrib.layers.fully_connected(inputs = hidden3, num_outputs = acts_data.shape[-1], activation_fn = None)

    loss = tf.losses.mean_squared_error(labels = output_action_true, predictions = output)
    # tf.summary.scalar('loss', loss)
    train = tf.train.AdamOptimizer().minimize(loss)

    # run the graph:
    epoch = 30
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        begin = time.time()
        # training phase
        for i in range(epoch):
            all_loss = []
            for batch_index in get_batch(obs_data, acts_data):
                obs_batch, acts_batch = obs_data[batch_index], acts_data[batch_index]
                res_loss, _ = sess.run([loss, train], feed_dict = {input_obs: obs_batch, output_action_true: acts_batch})
                all_loss.append(res_loss)
            all_loss = np.mean(all_loss)
            print('epoch {} -> loss {}'.format(i, res_loss))

        # env 
        env = gym.make(envname)
        max_steps = max_timestamps or env.spec.timestep_limit
         
        # predict phase 
        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset() # initial an observation
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(output, feed_dict = {input_obs: obs[None, :]}) # using network
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action) # return (observation, reward, done, info)
                totalr += r
                steps += 1
                # if render:
                #     env.render()
                if steps % 100 == 0:
                    print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns:', returns)
        print('mean return:', np.mean(returns))
        print('std of return:', np.std(returns))    
            
        end = time.time()
        print('total training time:{}'.format(end-begin))

if __name__ == '__main__':
    main()
    
# Result:
# epoch: 30
# num_rollouts: 20
# mean return: 9531.432405863372
# std of return: 2520.5479079319093
# total training time:283.80944776535034