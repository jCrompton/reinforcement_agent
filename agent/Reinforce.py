import time
import numpy as np
import random
import tensorflow as tf


class Agent:
    def __init__(self, env, shapes=[4, 16, 64, 256, 64, 32, 8, 2], rewards=2, dropout=0.6):
        self.env = env
        self.initial_state = env.reset()
        self.state_dim = env.reset().shape[0]
        self.flat_state = self.initial_state.reshape(self.initial_state.shape[0], -1)
        self.dim_action = env.action_space.n
        self.architecture = shapes
        self.rewards = 2
        self.parameters = {}
        self.activations = {}
        self.model_path = 'Train a model first'
        self.network_state = None
        self.network_action = None
        self.network_reward = None
        self.action_onehot = None
        self.keep_prob = None
        self.dropout = 0.6

    def create_placeholders(self):
        network_state = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='network_state')
        network_action = tf.placeholder(tf.int32, [None], name="network_action")
        network_reward = tf.placeholder(tf.float32,[None], name="network_reward")
        action_onehot = tf.one_hot(network_action, self.dim_action, name="actiononehot")
        keep_prob = tf.placeholder(tf.float32)
        return network_state, network_action, network_reward, action_onehot, keep_prob

    def initialize_parameters(self):
        for i in range(len(self.architecture) - 1):
            w_str = 'W{}'.format(i+1)
            b_str = 'b{}'.format(i+1)
            Wi = tf.get_variable(w_str, [self.architecture[i], self.architecture[i+1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            bi = tf.get_variable(b_str, [ self.architecture[i+1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.parameters[w_str] = Wi
            self.parameters[b_str] = bi  

    def forward_propagation(self, state, keep_prob):
        activations = {}
        b1 = self.parameters['b1']
        self.activations['Z1'] = tf.add(tf.matmul(state, self.parameters['W1']), b1)
        for i in range((len(self.parameters)/2) - 1):
            Wi = self.parameters['W{}'.format(i+2)]
            bi = self.parameters['b{}'.format(i+2)]
            Ai = tf.nn.relu(self.activations['Z{}'.format(i+1)])
            Ai_dropout = tf.nn.dropout(Ai, keep_prob)
            self.activations['Z{}'.format(i+2)] = tf.add(tf.matmul(Ai_dropout, Wi), bi)
        
        return self.activations['Z{}'.format((len(self.parameters)/2))]

    def linear_decay(self, epoch, limit, start):
        if epoch > limit:
            return 0
        return ((-start/limit)*epoch + start)

    def decay(self, epoch, total_frames):
        if epoch == 0:
            return 1.0
        return 1 - (0.5**((float(total_frames)/epoch) - 1)) 

    def initialize(self):
        self.initialize_parameters()

        network_state, network_action, network_reward, action_onehot, keep_prob = self.create_placeholders()
        reward_prection = self.forward_propagation(network_state, self.dropout)
        qreward = tf.reduce_sum(tf.multiply(reward_prection, action_onehot), reduction_indices = 1)
        loss = tf.reduce_mean(tf.square(network_reward - qreward))

        return network_state, network_action, network_reward, keep_prob, reward_prection, loss

    def train(self, epsilon = 0.8, BATCH_SIZE=32, GAMMA= 0.95, MAX_LEN_REPLAY_MEMORY = 30000,
              FRAMES_TO_PLAY = 300001, MIN_FRAMES_FOR_LEARNING = 1000, learning_rate=0.0001, print_cost=True, show_learning=False,
              persistent=False, epsilon_decay=False, cont=True, debug=False):
        
        tf.reset_default_graph()
        replay_memory = [] # (state, action, reward, terminalstate, state_t+1)
        observation = self.env.reset()
        n_actions = self.dim_action
        
        custom_architecture = self.architecture
                                                       
        n_x, _ = observation.reshape(observation.shape[0], -1).shape                  
        n_y = self.rewards                             
        costs = []                                        
        # Create Placeholders for network_state, network_action, reward and one_hot action
        # network_state, network_action, network_reward, action_onehot, keep_prob = self.create_placeholders()
        network_state, network_action, network_reward, keep_prob, reward_prection, loss = self.initialize()
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # Initialize all the variables
        init = tf.global_variables_initializer()
        with tf.Session() as sess:      
            # Run the initialization
            sess.run(init)
            for i_epoch in range(FRAMES_TO_PLAY):
                if epsilon_decay:
                    epsilon = self.linear_decay(i_epoch, FRAMES_TO_PLAY/2, 0.8)

                ### Select an action and perform this
                action = self.env.action_space.sample() 
                if len(replay_memory) > MIN_FRAMES_FOR_LEARNING and random.random() > epsilon:
                    pred_q = sess.run(reward_prection, feed_dict={network_state:[observation]})
                    action = np.argmax(pred_q)
                newobservation, reward, terminal, info = self.env.step(action)

                if show_learning:
                    self.env.render()
                ### Add the observation to our replay memory
                replay_memory.append((observation, action, reward, terminal, newobservation))

                ### Reset the environment if the agent died
                if terminal: 
                    newobservation = self.env.reset()
                observation = newobservation

                ### Learn once we have enough frames to start learning
                if len(replay_memory) > MIN_FRAMES_FOR_LEARNING:
                    experiences = random.sample(replay_memory, BATCH_SIZE)
                    totrain = [] # (state, action, delayed_reward)

                    ### Calculate the predicted reward
                    nextstates = [var[4] for var in experiences]
                    pred_reward = sess.run(reward_prection, feed_dict={network_state:nextstates, keep_prob:1.0})

                    ### On Policy update
                    for index in range(BATCH_SIZE):
                        # TODO: Pop state, action, reward, terminalstate, newstate out of experiences !
                        state, action, reward, terminalstate, newstate = experiences[index]
  
                        predicted_reward = max(pred_reward[index])

                        if terminalstate:
                            delayedreward = reward
                        else:
                            delayedreward = reward + GAMMA*predicted_reward
                        totrain.append((state, action, delayedreward))

                    ### Feed the train batch to the algorithm 
                    states = [var[0] for var in totrain]
                    actions = [var[1] for var in totrain]
                    rewards = [var[2] for var in totrain]

                    _, l= sess.run([optimizer, loss], feed_dict={network_state:states, network_action: actions, network_reward: rewards,
                                                                keep_prob: self.dropout})

                    ### If our memory is too big: remove the first element
                    if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                            replay_memory = replay_memory[1:]
                    if i_epoch%1000==1:
                        print("Epoch %d, loss: %f" % (i_epoch,l))
                    if i_epoch%100000==1 and persistent:
                        self.save(sess, i_epoch)
            if debug:
                for key in self.parameters.keys():
                    print('{} : {}'.format(key, self.parameters[key].eval()))
            self.save(sess, FRAMES_TO_PLAY)
            self.env.close()

    def save(self, sess, FRAMES_TO_PLAY):
        saver = tf.train.Saver()
        save_path = saver.save(sess, './models/model{}_{}'.format(len(self.architecture), FRAMES_TO_PLAY))
        self.model_path = save_path
        print('Model updated to {}'.format(save_path))

    def reinitialize(self, sess, debug=False):
        self.parameters = {}
        graph = tf.get_default_graph()
        self.initialize_parameters()

        network_state, network_action, network_reward, action_onehot, keep_prob = self.create_placeholders()

        saver = tf.train.Saver()
        saver.restore(sess, self.model_path)

        if debug:
            for key in self.parameters.keys():
                print('{} : {}'.format(key, self.parameters[key].eval()))

        predq = self.forward_propagation(network_state, self.dropout)

        return predq, network_state, keep_prob, saver

    def run(self, print_time=True, debug=False, epsilon=0.05):
        tf.reset_default_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            predq, network_state, keep_prob, saver = self.reinitialize(sess, debug=debug)
            sess.run(init)
            observation = self.env.reset()
            term = False
            start_time = time.time()
            while not term:
                obs = self.env.render()
                action = self.env.action_space.sample()
                if random.random() > epsilon:
                    pred_q = sess.run(predq, feed_dict={network_state:[observation], keep_prob: 1.0})
                    action = np.argmax(pred_q)
                observation, _, term, _ = self.env.step(action)
            if print_time:
                print('{} seconds elapsed'.format(time.time()-start_time))

    def random_run(self, print_time=True):
        self.env.reset()
        term = False
        start_time = time.time()
        while not term:
            obs = self.env.render()
            action = self.env.action_space.sample()
            _, _, term, _ = self.env.step(action)
        if print_time:
            print('{} seconds elapsed'.format(time.time()-start_time))
