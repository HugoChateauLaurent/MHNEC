import pickle
import os
from . import utils, parameters
import numpy as np
import shutil
from itertools import chain, combinations

import torch
from tqdm import tqdm

from .agents import NECAgent
from .envs import AtariEnv
from .memory import ExperienceReplay

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AbstractExperiment(object):

    def __init__(self, args):
        super(AbstractExperiment, self).__init__()

        self.params = dict({})
        self.exploration = dict({})
        self.name = type(self).__name__+'_'+utils.formatted_time() if args.experiment_id is None else args.experiment_id
        print(self.name)

        self.set_default_parameters()
        self.set_args_parameters(args)

        # For details: https://www.plafrim.fr/
        if self.params["use_plafrim"].default_value:
            from . import plafrim_utils
            self.job_dispatcher = plafrim_utils.JobDispatcher(self.name)

    def add_parameter(self, name, default_value, comment=None):
        assert name not in self.params.keys()
        self.params[name] = parameters.Parameter(name, default_value, comment)

    def set_default_parameters(self):

        self.add_parameter('id', 'default', comment='Experiment ID')
        self.add_parameter('seed', 123, comment='Random seed')
        self.add_parameter('game', 'pong', comment='ATARI game')
        self.add_parameter('T_max', int(1e7), comment='Number of training steps (4x number of frames)')
        self.add_parameter('max_episode_length', int(108e3), comment='Max episode length (0 to disable)')
        self.add_parameter('history_length', 4, comment='Number of consecutive states processed (ATARI)')  # 1 for MFEC (originally), 4 for MFEC (in NEC paper) and NEC
        self.add_parameter('algorithm', 'NEC', comment='Algorithm')
        self.add_parameter('hidden_size', 512, comment='Hidden size')
        self.add_parameter('key_size', 128, comment='Key size')  # 64 for MFEC, 128 for NEC
        self.add_parameter('num_neighbours', 50, comment='Number of nearest neighbours')  # 11 for MFEC, 50 for NEC
        self.add_parameter('separation_beta', 1, comment='Separation Beta')
        self.add_parameter('memory_capacity', int(1e5), comment='Experience replay memory capacity')
        self.add_parameter('dictionary_capacity', 500)#int(5e5), comment='Dictionary capacity (per action)')  # 1e6 for MFEC, 5e5 for NEC
        print("WARNING: changed dictionary_capacity")
        self.add_parameter('replay_frequency', 4, comment='Frequency of sampling from memory')
        self.add_parameter('episodic_multi_step', 100, comment='Number of steps for multi-step return from end of episode')  # Infinity for MFEC, 100 for NEC
        self.add_parameter('epsilon_initial', 1, comment='Initial value of ε-greedy policy')
        self.add_parameter('epsilon_final', 0.001, comment='Final value of ε-greedy policy')  # 0.005 for MFEC, 0.001 for NEC
        self.add_parameter('epsilon_anneal_start', 5000, comment='Number of steps before annealing ε')
        self.add_parameter('epsilon_anneal_end', 25000, comment='Number of steps to finish annealing ε')
        self.add_parameter('discount', .99, comment='Discount factor')  # 1 for MFEC, 0.99 for NEC
        self.add_parameter('learning_rate', 7.92468721e-6, comment='Network learning rate')
        self.add_parameter('rmsprop_decay', 0.95, comment='RMSprop decay')
        self.add_parameter('rmsprop_epsilon', 0.01, comment='RMSprop epsilon')
        self.add_parameter('rmsprop_momentum', 0, comment='RMSprop momentum')
        self.add_parameter('dictionary_learning_rate', 0.1, comment='Dictionary learning rate')
        self.add_parameter('kernel', 'mean_IDW', comment='Kernel function')  # mean for MFEC, mean_IDW for NEC
        self.add_parameter('kernel_delta', 1e-3, comment='Mean IDW kernel delta')
        self.add_parameter('batch_size', 32, comment='Batch size')
        self.add_parameter('learn_start', 3000)#50000, comment='Number of steps before starting training')  # 0 for MFEC, 50000 for NEC
        print("WARNING: changed learn_start")
        self.add_parameter('evaluation_interval', 1000, comment='Number of training steps between evaluations')
        self.add_parameter('evaluation_episodes', 10, comment='Number of evaluation episodes to average over')
        self.add_parameter('evaluation_size', 500, comment='Number of transitions to use for validating Q')
        self.add_parameter('evaluation_epsilon', 0, comment='Value of ε-greedy policy for evaluation')
        self.add_parameter('checkpoint_interval', 0, comment='Number of training steps between saving buffers (0 to disable)')
        self.add_parameter('render', False, comment='Display screen (testing only)')

    def set_args_parameters(self, args):
        self.add_parameter('args', args, comment='Keep track of all arguments that were passed for reproducibility purposes')
        self.add_parameter('use_plafrim', args.plafrim, comment='Whether or not to use the PlaFRIM computing platform')

    def __call__(self):

        path = 'logs/'+self.name
        os.makedirs(path+"/code")
        shutil.copy("main.py", path+"/code")
        shutil.copytree("MHNEC", path+"/code"+"/MHNEC")

        self.setup_exploration()
        self.launch_exploration()

    def setup_exploration(self):
        raise NotImplementedError("Subclasses should implement this!")

    def launch_exploration(self):
        default_params = {p for p in self.params.values() if p.default} # params that will not change
        explorer_params = set(self.params.values()) - default_params # params that will change

        # Compute the number of simulations in the exploration
        if len(explorer_params) > 0:
            exploration_length = [len(p.exploration_values) for p in explorer_params]
            assert np.all(np.array(exploration_length) == exploration_length[0]) # check if all explorer parameters have the same number of exploration values
            exploration_length = exploration_length[0]
        else:
            exploration_length = 1

        iterator = range(exploration_length)
        iterator = tqdm(iterator)
        for i in iterator:
            sim_params = dict({}) # Now get parameters for this simulation
            for p in explorer_params:
                sim_params[p.name] = p.exploration_values[i]
            for p in default_params:
                sim_params[p.name] = p.default_value

            # run this simulation
            self.launch_simulation(sim_params, i)

        # For details: https://www.plafrim.fr/
        if self.params["use_plafrim"].default_value:
            self.job_dispatcher.launch_jobs()


    def launch_simulation(self, sim_params, sim_id):

        sim_params["rng"] = np.random.default_rng(sim_params["seed"])
        sim_params["xp_cls"] = self.__class__

        path = 'logs/'+self.name+'/simulations/sim_'+'{0:07d}'.format(sim_id)+'_' + utils.formatted_time()
        os.makedirs(path)
        with open(path+"/params.pickle", "wb") as f:
            pickle.dump(sim_params, f)

        # For details: https://www.plafrim.fr/
        if sim_params["use_plafrim"]:
            from . import plafrim_utils
            self.job_dispatcher.tasks.append(plafrim_utils.Task('logs/'+self.name+'/code', path))
        else:
            self.simulate(path, sim_params)

    @staticmethod
    def simulate(path, sim_params):

        args = AttrDict(sim_params)

        # Setup
        np.random.seed(sim_params["seed"])
        torch.manual_seed(sim_params["seed"])
        if torch.cuda.is_available():
            args.device = torch.device('cuda')
            torch.cuda.manual_seed(sim_params["seed"])
        else:
            args.device = torch.device('cpu')
        metrics = {'train_steps': [], 'train_episodes': [], 'train_rewards': [], 'test_steps': [], 'test_rewards': [], 'test_Qs': []}

        # Environment
        env = AtariEnv(args)
        env.train()


        # Agent and memory
        if args.algorithm == 'MFEC':
            agent = MFECAgent(args, env.observation_space.shape, env.action_space.n, env.hash_space.shape[0])
        elif args.algorithm == 'NEC':
            agent = NECAgent(args, env.observation_space.shape, env.action_space.n, env.hash_space.shape[0])
            mem = ExperienceReplay(args.memory_capacity, env.observation_space.shape, args.device)

        # Construct validation memory
        val_mem = ExperienceReplay(args.evaluation_size, env.observation_space.shape, args.device)
        T, done, states = 0, True, []    # Store transition data in episodic buffers
        while T < args.evaluation_size:
            if done:
                state, done = env.reset(), False
            states.append(state.cpu().numpy())    # Append transition data to episodic buffers
            state, _, done = env.step(env.action_space.sample())
            T += 1
        val_mem.append_batch(np.stack(states), np.zeros((args.evaluation_size, ), dtype=np.int64), np.zeros((args.evaluation_size, ), dtype=np.float32))


        # Training loop
        agent.train()
        T, done, epsilon = 0, True, args.epsilon_initial
        agent.set_epsilon(epsilon)
        for T in tqdm(range(1, args.T_max + 1)):
            if done:
                state, done = env.reset(), False
                states, actions, rewards, keys, values, hashes = [], [], [], [], [], []    # Store transition data in episodic buffers

            # Linearly anneal ε over set interval
            if T > args.epsilon_anneal_start and T <= args.epsilon_anneal_end:
                epsilon -= (args.epsilon_initial - args.epsilon_final) / (args.epsilon_anneal_end - args.epsilon_anneal_start)
                agent.set_epsilon(epsilon)
            
            # Append transition data to episodic buffers (1/2)
            states.append(state.cpu().numpy())
            hashes.append(env.get_state_hash())    # Use environment state hash function
            
            # Choose an action according to the policy
            action, key, value = agent.act(state, return_key_value=True)
            state, reward, done = env.step(action)    # Step

            # Append transition data to episodic buffers (2/2); note that original NEC implementation does not recalculate keys/values at the end of the episode
            actions.append(action)
            rewards.append(reward)
            keys.append(key.cpu().numpy())
            values.append(value)

            # Calculate returns at episode to batch memory updates
            if done:
                episode_T = len(rewards)
                returns, multistep_returns = [None] * episode_T, [None] * episode_T
                returns.append(0)
                for i in range(episode_T - 1, -1, -1):    # Calculate return-to-go in reverse
                    returns[i] = rewards[i] + args.discount * returns[i + 1]
                    if episode_T - i > args.episodic_multi_step:    # Calculate multi-step returns (originally only for NEC)
                        multistep_returns[i] = returns[i] + args.discount ** args.episodic_multi_step * (values[i + args.episodic_multi_step] - returns[i + args.episodic_multi_step])
                    else:    # Calculate Monte Carlo returns (originally only for MFEC)
                        multistep_returns[i] = returns[i]
                states, actions, returns, keys, hashes = np.stack(states), np.asarray(actions, dtype=np.int64), np.asarray(multistep_returns, dtype=np.float32), np.stack(keys), np.stack(hashes)
                unique_actions, unique_action_reverse_idxs = np.unique(actions, return_inverse=True)    # Find every unique action taken and indices
                for i, a in enumerate(unique_actions):
                    a_idxs = (unique_action_reverse_idxs == i).nonzero()[0]
                    agent.update_memory_batch(a.item(), keys[a_idxs], returns[a_idxs][:, np.newaxis], hashes[a_idxs])    # Append transition to DND of action in batch
                if args.algorithm == 'NEC':
                    mem.append_batch(states, actions, returns)    # Append transition to memory in batch

                # Save metrics
                metrics['train_steps'].append(T)
                metrics['train_episodes'].append(len(metrics['train_episodes']) + 1)
                metrics['train_rewards'].append(sum(rewards))
                # torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

            # Train and test
            if T >= args.learn_start:
                if args.algorithm == 'NEC' and T % args.replay_frequency == 0:
                    agent.learn(mem)    # Train network
                
                if T % args.evaluation_interval == 0:
                    agent.eval()    # Set agent to evaluation mode
                    test_rewards, test_Qs = test(args, T, agent, val_mem, results_dir)    # Test
                    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(sum(test_rewards) / args.evaluation_episodes) + ' | Avg. Q: ' + str(sum(test_Qs) / args.evaluation_size))
                    agent.train()    # Set agent back to training mode
                    metrics['test_steps'].append(T)
                    metrics['test_rewards'].append(test_rewards)
                    metrics['test_Qs'].append(test_Qs)
                    # torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        env.close()

        

        utils.save_results(path, results)


class DefaultExperiment(AbstractExperiment):
    def __init__(self, args):
        super(DefaultExperiment, self).__init__(args)
    def setup_exploration(self):
        pass

class DebugExperiment(AbstractExperiment):
    def __init__(self, args):
        super(DebugExperiment, self).__init__(args)
    def setup_exploration(self):
        # self.params["env"].default_value = lambda rng: environments.NavawongseEnvironment(n_contexts=2, n_tasks=100, rng=rng)
        # self.params["N_scale"].default_value = .001
        for m in [None]:
            for use_mhn in [0]:#,1]:
                for seed in range(50,55):
                    self.params["modulations"].exploration_values.append(m)
                    self.params["use_mhn"].exploration_values.append(use_mhn)
                    self.params["seed"].exploration_values.append(seed)

class RandomParametersExperiment(AbstractExperiment):
    def setup_exploration(self):
        rng = np.random.default_rng(42) # rng for generating reproducible series of random seeds and parameters
        

        N_seeds = 300
        
        for i in range(N_seeds):#250):

            pass