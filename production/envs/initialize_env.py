import datetime
import random
from collections import deque
import sys
import os

from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.transport import *
from production.envs.machine import *
from production.envs.sink import *
from production.envs.source import *
from production.envs.production_env import *
import numpy as np
import pandas as pd
import statistics
import datetime as dt
from collections import Counter
from collections import defaultdict

PRINT_CONSOLE = False
EPSILON = 0.000001
EXPORT_FREQUENCY = 10 ** 3
EXPORT_NO_LOGS = False

PATH_TIME = "log/" + datetime.now().strftime("%Y%m%d_%H%M%S")


def define_production_parameters(env, episode):
    """
    Describe production system parameters
    """
    parameters = dict()

    # fix seed per episode group for reproducibility
    stable_seed = 42 + (episode // 10)
    np.random.seed(stable_seed)
    random.seed(stable_seed)
    parameters.update({'RANDOM_SEED': stable_seed})

    parameters.update({'episode': episode})
    parameters.update({'NUM_ORDERS': 10 ** 8})
    parameters.update({'time_end': 0.0})
    parameters.update({'stop_criteria': env.event()})
    parameters.update({'step_criteria': env.event()})
    parameters.update({'continue_criteria': env.event()})
    parameters.update({'SMOOTHING_RATE': 0.9})
    parameters.update({'EPSILON': 0.0001})
    parameters.update({'PRINT_CONSOLE': False})
    parameters.update({'EXPORT_FREQUENCY': 1000})
    parameters.update({'CONWIP_ORDER_LIMIT': 20})
    parameters.update({'CHANGE_SCENARIO_AFTER_EPISODES': 100000})

    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    parameters.update({'PATH_TIME': "logs/machine_shop_" + str(datetime.now().strftime("%Y%m%d_%H%M%S"))})
    parameters.update({'EXPORT_NO_LOGS': True})
    parameters.update({'EXPORT_OVERVIEWS': False})
    parameters.update({'EXPORT_DETAILS': False})
    parameters.update({'EXPORT_INDIVIDUAL_OVERVIEWS': False})
    parameters = extend_agent_parameters(parameters)
    parameters = extend_production_parameters(parameters)
    return parameters


def extend_agent_parameters(parameters):
    parameters.update({'TRANSP_AGENT_TYPE': "TRPO"})
    parameters.update({'TRANSP_AGENT_STATE': ['sys_buffer_state', 'order_state']})
    parameters.update({'TRANSP_AGENT_REWARD': "weighted_objectives"})
    parameters.update({'TRANSP_AGENT_REWARD_SPARSE': ""})
    parameters.update({'TRANSP_AGENT_REWARD_EPISODE_LIMIT': 0})
    parameters.update({'TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE': "valid"})
    parameters.update({'TRANSP_AGENT_REWARD_SUBSET_WEIGHTS': [1.0, 1.0]})
    parameters.update({'TRANSP_AGENT_REWARD_OBJECTIVE_WEIGHTS': {'order_reward': 1, 'mach_cost': 1.2}})
    parameters.update({'TRANSP_AGENT_REWARD_WAITING_ACTION': 0.0})
    parameters.update({'TRANSP_AGENT_REWARD_INVALID_ACTION': -2})
    parameters.update({'TRANSP_AGENT_MAX_INVALID_ACTIONS': 3})
    parameters.update({'TRANSP_AGENT_WAITING_TIME_ACTION': 2})
    parameters.update({'TRANSP_AGENT_ACTION_MAPPING': 'direct'})
    parameters.update({'TRANSP_AGENT_WAITING_ACTION': False})
    parameters.update({'TRANSP_AGENT_EMPTY_ACTION': False})
    parameters.update({'TRANSP_AGENT_CONWIP_INV': 15})
    parameters.update({'WAITING_TIME_THRESHOLD': 1000})
    # GA parameters
    parameters.update({'USE_GA': False})
    parameters.update({'GA_SELECTION_RATIO': 0.65})
    parameters.update({'GA_MIN_SELECTED_ORDERS': 2})
    parameters.update({'GA_MAX_SELECTED_ORDERS': 8})
    parameters.update({'GA_CACHE_UPDATE_INTERVAL': 5})
    # SCPR mode: greedy top-k ranking without GA crossover/mutation (set True to enable)
    parameters.update({'USE_SCPR_ONLY': False})

    return parameters


def extend_production_parameters(parameters):
    parameters.update({'NUM_TRANSP_AGENTS': 1})
    parameters.update({'NUM_MACHINES': 12})
    parameters.update({'NUM_MACHINES_INIT': 12})
    parameters.update({'NUM_SOURCES': 1})
    parameters.update({'NUM_SINKS': 1})
    parameters.update({'NUM_RESOURCES': parameters['NUM_MACHINES'] + parameters['NUM_SOURCES'] + parameters['NUM_SINKS']})
    parameters.update({'NUM_PROD_VARIANTS': 1})
    parameters.update({'NUM_PROD_STEPS': 7})
    # Transport parameters
    parameters.update({'TRANSP_SPEED': 1.0 * 60.0})
    parameters.update({'RESP_AREA_TRANSP': [[[True for i in range(parameters['NUM_RESOURCES'])] for j in range(parameters['NUM_RESOURCES'])] for k in range(parameters['NUM_TRANSP_AGENTS'])]})
    # Source parameters
    parameters.update({'SOURCE_CAPACITIES': [3] * parameters['NUM_SOURCES']})
    parameters.update({'RESP_AREA_SOURCE': [[3, 5, 8]]})
    parameters.update({'MTOG': [30.0, 15.0, 15.0]})
    parameters.update({'SOURCE_ORDER_GENERATION_TYPE': "ALWAYS_FILL_UP"})
    # Machine parameters
    parameters.update({'MACHINE_AGENT_TYPE': "FIFO"})
    parameters.update({'MACHINE_COST': [3, 1, 4, 1.1]})
    parameters.update({'MACHINE_GROUPS': [0, 1, 2, 3, 4, 5, 6, 1, 2, 4, 5, 6]})
    parameters.update({'RESP_AREA_MACHINE': [[4, 9], [4], [7, 12], [0, 7, 12], [5, 6, 10, 11], [1], [1], [1], [0, 7, 12], [5, 6, 10, 11], [1], [1], [1]]})
    parameters.update({'MACHINE_TYPE': [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]})
    parameters.update({'MACHINE_GROUPS': [0, 1, 2, 3, 4, 5, 6, 1, 2, 4, 5, 6]})
    parameters.update({'MACHINES_PROCESSING_TIME': [9]})
    parameters.update({'MACHINE_CAPACITIES': [8] * parameters['NUM_MACHINES']})
    parameters.update({'MIN_PROCESS_TIME': [15.0] * parameters['NUM_MACHINES']})
    parameters.update({'AVERAGE_PROCESS_TIME': [120.0] * parameters['NUM_MACHINES']})
    parameters.update({'MAX_PROCESS_TIME': [250.0] * parameters['NUM_MACHINES']})
    parameters.update({'CHANGEOVER_TIME': 0.0})
    parameters.update({'MTBF': [1000.0] * parameters['NUM_MACHINES']})
    parameters.update({'MTOL': [30.0] * parameters['NUM_MACHINES']})
    # Order parameters
    parameters.update({'ORDER_DISTRIBUTION': [1.0 / parameters['NUM_MACHINES']] * parameters['NUM_MACHINES']})
    parameters.update({'VARIANT_DISTRIBUTION': [1.0 / parameters['NUM_PROD_VARIANTS']] * parameters['NUM_PROD_VARIANTS']})
    # Handling time
    parameters.update({'TIME_TO_LOAD_MACHINE': 60.0 / 60.0})
    parameters.update({'TIME_TO_UNLOAD_MACHINE': 60.0 / 60.0})
    parameters.update({'TIME_TO_LOAD_SOURCE': 60.0 / 60.0})
    parameters.update({'TIME_TO_UNLOAD_SOURCE': 60.0 / 60.0})
    # Transport time matrix
    parameters.update({'TRANSP_DISTANCE': [[50.0 for x in range(parameters['NUM_RESOURCES'])] for y in range(parameters['NUM_RESOURCES'])]})
    parameters.update({'TRANSP_TIME': [[0.0 for x in range(parameters['NUM_RESOURCES'])] for y in range(parameters['NUM_RESOURCES'])]})
    for i in range(parameters['NUM_RESOURCES']):
        for j in range(parameters['NUM_RESOURCES']):
            parameters['TRANSP_TIME'][i][j] = parameters['TRANSP_DISTANCE'][i][j] / parameters['TRANSP_SPEED']
            if i == j:
                parameters['TRANSP_TIME'][i][j] = 0.0
    parameters.update({'MAX_TRANSP_TIME': np.array(parameters['TRANSP_TIME']).max()})

    return parameters


def define_production_statistics(parameters):
    statistics = dict()
    stat_episode = dict()
    statistics.update({'stat_machines_working': np.array([0.0] * parameters['NUM_MACHINES'])})
    statistics.update({'stat_machines_broken': np.array([0.0] * parameters['NUM_MACHINES'])})
    statistics.update({'stat_machines_idle': np.array([0.0] * parameters['NUM_MACHINES'])})
    statistics.update({'stat_machines_changeover': np.array([0.0] * parameters['NUM_MACHINES'])})
    statistics.update({'stat_machines_processed_orders': np.array([0.0] * parameters['NUM_MACHINES'])})
    statistics.update({'stat_machines': [statistics['stat_machines_working'], statistics['stat_machines_broken'],
                                         statistics['stat_machines_idle'], statistics['stat_machines_changeover']]})
    statistics.update({'stat_transp_working': np.array([0.0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp_walking': np.array([0.0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp_handling': np.array([0.0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp_idle': np.array([0.0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp': [statistics['stat_transp_walking'], statistics['stat_transp_idle']]})
    statistics.update({'stat_transp_selected_idle': np.array([0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp_forced_idle': np.array([0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_transp_threshold_waiting_reached': np.array([0] * parameters['NUM_TRANSP_AGENTS'])})
    statistics.update({'stat_order_sop': defaultdict(int)})
    statistics.update({'stat_order_eop': defaultdict(int)})
    statistics.update({'stat_order_waiting': defaultdict(int)})
    statistics.update({'stat_order_processing': defaultdict(int)})
    statistics.update({'stat_order_handling': defaultdict(int)})
    statistics.update({'stat_order_leadtime': defaultdict(int)})
    statistics.update(
        {'stat_order': [statistics['stat_order_sop'], statistics['stat_order_eop'], statistics['stat_order_waiting'],
                        statistics['stat_order_processing'], statistics['stat_order_handling'],
                        statistics['stat_order_leadtime']]})
    statistics.update({'stat_inv_buffer_in': np.array([0.0] * parameters['NUM_RESOURCES'])})
    statistics.update({'stat_inv_buffer_out': np.array([0.0] * parameters['NUM_RESOURCES'])})
    statistics.update({'stat_inv_buffer_in_mean': [np.array([0.0] * parameters['NUM_RESOURCES']),
                                                   np.array([0.0] * parameters['NUM_RESOURCES'])]})
    statistics.update({'stat_inv_buffer_out_mean': [np.array([0.0] * parameters['NUM_RESOURCES']),
                                                    np.array([0.0] * parameters['NUM_RESOURCES'])]})
    statistics.update({'stat_inv': [statistics['stat_inv_buffer_in'], statistics['stat_inv_buffer_out']]})
    statistics.update({'stat_inv_episode': [[0.0, 0]]})
    statistics.update({'stat_agent_reward': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]})

    if not parameters.get('EXPORT_NO_LOGS', False):
        statistics.update({'agent_reward_log': open(parameters['PATH_TIME'] + "_agent_reward_log.txt", "w")})
        statistics.update({'episode_log': open(parameters['PATH_TIME'] + "_episode_log.txt", "w")})
    else:
        class DummyFile:
            def write(self, *args): pass
            def close(self): pass
        statistics.update({'agent_reward_log': DummyFile()})
        statistics.update({'episode_log': DummyFile()})

    statistics.update({'episode_statistics': ['stat_machines_working', 'stat_machines_changeover',
                                              'stat_machines_broken', 'stat_machines_idle', 'stat_machines_processed_orders',
                                              'stat_transp_working', 'stat_transp_walking', 'stat_transp_handling',
                                              'stat_transp_idle']})
    statistics.update({'sim_start_time': ""})
    statistics.update({'sim_end_time': ""})
    statistics.update({'stat_prefilled_orders': ""})
    statistics.update({'episode_log_header': ['episode_counter', 'sim_step', 'sim_time', 'dt', 'dt_real_time', 'valid_actions', 'total_reward', 'machines_working', 'machines_changeover', 'machines_broken', 'machines_idle', 'processed_orders', 'transp_working', 'transp_walking', 'transp_handling', 'transp_idle', 'machines_total', 'selected_idle', 'forced_idle', 'threshold_waiting', 'finished_orders', 'order_waiting_time', 'alpha', 'inventory']})
    string = ""
    for x in statistics['episode_log_header']:
        string = string + x + ","
    string = string[:-1]
    statistics['episode_log'].write("%s\n" % (string))
    string = "episode,sim_step,sim_time,action,reward,action_valid,state"
    statistics['agent_reward_log'].write("%s\n" % (string))
    statistics['agent_reward_log'].close()
    for stat in statistics['episode_statistics']:
        stat_episode.update({stat: np.array([0.0] * len(statistics[stat]))})
    statistics.update({'orders_done': deque()})
    return statistics, stat_episode


def define_production_resources(env, statistics, parameters, agents, time_calc):
    resources = dict()
    resources.update({'sources': [Source(env=env, id=i, capacity=parameters['SOURCE_CAPACITIES'][i],
                                         resp_area=parameters['RESP_AREA_SOURCE'][i],
                                         statistics=statistics, parameters=parameters, resources=resources,
                                         agents=agents, time_calc=time_calc,
                                         location=None, label=None)
                                  for i in range(parameters['NUM_SOURCES'])]})
    resources.update({'sinks': [Sink(env=env, id=i + parameters['NUM_SOURCES'],
                                     statistics=statistics, parameters=parameters, resources=resources, agents=agents,
                                     time_calc=time_calc,
                                     location=None, label=None)
                                for i in range(parameters['NUM_SINKS'])]})
    resources.update({'machines': [Machine(env=env, id=i + parameters['NUM_SINKS'] + parameters['NUM_SOURCES'], capacity=parameters['MACHINE_CAPACITIES'][i],
                     agent_type=parameters['MACHINE_AGENT_TYPE'], resp_area=parameters['RESP_AREA_MACHINE'][i],
                     machine_group=parameters['MACHINE_GROUPS'][i], machine_type=parameters['MACHINE_TYPE'][i],
                     statistics=statistics, parameters=parameters, resources=resources, agents=agents, time_calc=time_calc,
                     location=None, label=None)
                        for i in range(parameters['NUM_MACHINES'])]})

    for group in [1, 2, 4, 5, 6]:
        new_machines = []
        resources.update({f'machine_group_{group}': new_machines})

    temp_resources = []
    temp_resources.extend(resources['sources'])
    temp_resources.extend(resources['sinks'])
    temp_resources.extend(resources['machines'])

    for group in [1, 2, 4, 5, 6]:
        temp_resources.extend(resources.get(f'machine_group_{group}', []))

    # Create source and machine normalizers
    source_wt_normalizer = ZScoreNormalization('exp', alpha=0.01)
    machine_wt_normalizer = ZScoreNormalization('exp', alpha=0.01)
    for mach in resources['machines']:
        mach.machine_wt_normalizer = machine_wt_normalizer
    for sourc in resources['sources']:
        sourc.source_wt_normalizer = source_wt_normalizer
    resources.update({'all_resources': temp_resources})

    resources.update({'transps': [Transport(env=env, id=i, resp_area=parameters['RESP_AREA_TRANSP'][i],
                                                  agent_type=parameters['TRANSP_AGENT_TYPE'],
                                                  statistics=statistics, parameters=parameters, resources=resources,
                                                  agents=agents, time_calc=time_calc, location=None, label=None)
                                        for i in range(parameters['NUM_TRANSP_AGENTS'])]})

    resources.update({'repairman': simpy.PreemptiveResource(env, capacity=parameters['NUM_MACHINES_INIT'] - 1)})

    env.process(other_jobs(env, resources['repairman']))

    print("All resources types: ", [x.type for x in resources['all_resources']])
    print("All resources ids: ", [x.id for x in resources['all_resources']])
    print("Number of sources: ", len(resources['sources']))
    print("Number of sinks: ", len(resources['sinks']))
    print("Number of machines: ", len(resources['machines']))
    return resources
