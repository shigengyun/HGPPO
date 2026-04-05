import sys
from itertools import chain
import gymnasium as gym
import simpy
from gymnasium import spaces
from logger import *
import numpy as np
from production.envs.initialize_env import *
from production.envs.machine import Machine
from production.envs.resources import *
from production.envs.time_calc import Time_calc, ZScoreNormalization
from datetime import datetime
from production.envs.logging_config import setup_logging, restore_logging
from production.envs.transport import Transport
from production.envs.heuristics import Decision_Heuristic


class ProductionEnv(gym.Env):
    def __init__(self, max_episode_timesteps, **kwargs):
        super(ProductionEnv, self).__init__()
        self.count_episode = 0
        self.last_export_time = 0.0
        self.last_export_real_time = datetime.now()
        self.env = simpy.Environment()
        self.counter = 0
        self.agents = dict()
        self.parameters = define_production_parameters(env=self.env, episode=self.count_episode)
        self.time_calc = Time_calc(parameters=self.parameters, episode=self.count_episode)
        self.statistics, self.stat_episode = define_production_statistics(parameters=self.parameters)
        self.resources = define_production_resources(env=self.env, statistics=self.statistics, parameters=self.parameters, agents=self.agents, time_calc=self.time_calc)
        self.statistics['sim_start_time'] = datetime.now()
        self.states1 = None
        self.observation_space = self._define_observation_space()
        self.action_space = self._define_action_space()
        self.max_episode_timesteps = max_episode_timesteps

    def step(self, actions):
        self.counter += 1
        reward = 0
        done = False
        info = {}
        truncated = False
        print(self.counter, "Agent-Action: ", str(actions))
        if self.counter == self.max_episode_timesteps:
            print("Last episode action ", datetime.now())
            done = True
        for agent in self.resources['transps'][0].agents_waiting_for_action:
            agent = self.resources['transps'][0].agents_waiting_for_action.pop(0)
            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                agent.next_action = [int(actions)]
                print("Agent-Action: ", str(agent.next_action))
            agent.state_before = None
            self.parameters['continue_criteria'].succeed()
            self.parameters['continue_criteria'] = self.env.event()
            self.env.run(until=self.parameters['step_criteria'])
            reward, done = agent.calculate_reward(actions)
            if done:
                print("Last episode action ", datetime.now())
            agent = Transport.agents_waiting_for_action[0]
            states = agent.calculate_state()
            states = np.array(states, dtype=np.float32)
            print("states:", states)
            if self.states1 is None or not np.array_equal(self.states1, states):
                print("state changed")
            self.states1 = states
            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions)]
            elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions[0]), int(actions[1])]
            self.statistics['stat_agent_reward'][-1][4] = round(reward, 5)
            self.statistics['stat_agent_reward'][-1][5] = agent.next_action_valid
            self.statistics['stat_agent_reward'].append([self.count_episode, self.counter, round(self.env.now, 5), None, None, None, states])
            print("number of sinks orders:", len(self.resources['sinks'][0].buffer_in_indiv))
            return states, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print("####### Reset Environment #######")
        self.count_episode += 1
        self.counter = 0
        print("Sim start time: ", self.statistics['sim_start_time'])
        if self.env.now == 0.0:
            print('Run machine shop simpy environment')
            self.env.run(until=self.parameters['step_criteria'])
        states = self.resources['transps'][0].calculate_state()
        states = np.array(states, dtype=np.float32)
        print("states:", states)
        return states, {}

    def close(self):
        print("####### Close Environment #######")
        if not self.parameters['EXPORT_NO_LOGS']:
            self.statistics.update({'time_end': self.env.now})
            export_statistics_logging(statistics=self.statistics, parameters=self.parameters, resources=self.resources)
        super().close()

    def render(self, mode='human', close=False):
        print("####### Render Environment #######")
        pass

    def _define_observation_space(self):
        total_dims = 0
        # Base state: 13 dims (1 source + 12 machines)
        basic_state_dims = len(self.resources['sources']) + len(self.resources['machines'])
        total_dims += basic_state_dims
        agent_state_config = self.parameters.get('TRANSP_AGENT_STATE', [])
        if 'sys_buffer_state' in agent_state_config:
            # Station load info: 13 dims
            total_dims += basic_state_dims
        if 'order_state' in agent_state_config:
            # Global order info: 3 dims (avg wait, avg progress, avg remaining time)
            total_dims += 3
        if 'ga_selection_state' in agent_state_config and self.parameters.get('USE_GA', False):
            # GA screening info: 13 dims + 3 global dims
            total_dims += basic_state_dims
            total_dims += 3
        print(f"Observation space dimensions: {total_dims}")
        return spaces.Box(low=0.0, high=1.0, shape=(total_dims,), dtype=np.float32)

    def _define_action_space(self):
        number = len(self.resources['transps'][0].mapping)
        return spaces.Discrete(number)
