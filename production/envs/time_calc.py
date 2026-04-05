import numpy as np
from collections import deque
from production.envs.sink import *
from production.envs.Job_ProcessTime import process_times
class Time_calc:
    def __init__(self, parameters, episode):
        self.parmeters = parameters

        """Random Seed for random numbers"""
        np.random.seed(parameters['RANDOM_SEED'] + episode)
        self.randomStreams = {
            "process_time": [np.random.RandomState(np.random.randint(100)) for i in range(parameters['NUM_MACHINES'])],
            "machine_failure": [np.random.RandomState(np.random.randint(100)) for i in
                                range(parameters['NUM_MACHINES'])],
            "repair_time": [np.random.RandomState(np.random.randint(100)) for i in range(parameters['NUM_MACHINES'])],
            "order_generation": [np.random.RandomState(np.random.randint(100)) for i in
                                 range(parameters['NUM_SOURCES'])],
            "order_sequence": np.random.RandomState(np.random.randint(100)),
            "transp_agent": [np.random.RandomState(np.random.randint(100)) for i in
                             range(parameters['NUM_TRANSP_AGENTS'])],
            "filled_initial_system": np.random.RandomState(np.random.randint(100))}
    """
    Utility procedures
    """
    def get_inventory_level(self, statistics):
        return statistics['stat_inv_episode'][-1][1]
    # Return current processing time
    def processing_time(self,parameters, order):
        machine_id=order.current_location.machine_group
        # Randomly select a row from process_times dataset and retrieve the processing time for the given machine_id
        random_row_index = self.randomStreams["process_time"][machine_id].randint(0, len(process_times))
        result_time = process_times[random_row_index][machine_id]
        return result_time

    def next_step_processing_time(self, parameters, order):
        """
        Calculate processing time for the next step of an order.
        """
        if order.get_next_step().type == 'sink':
            return 1  # Set minimum processing time when next step is sink
        machine_id = order.get_next_step().machine_group
        # Randomly select a row from process_times dataset and retrieve the processing time for the given machine_id
        random_row_index = self.randomStreams["process_time"][machine_id].randint(0, len(process_times))
        result_time = process_times[random_row_index][machine_id]
        return result_time

    def remaining_steps_processing_time(self,order):
        """Calculate total processing time for all remaining steps of an order."""
        total_time = 0
        original_actual_step = order.actual_step  # Save current step index
        while order.get_next_step().type != 'sink':
            if order.current_location.type=='source':
                return self.average_remaining_waiting_time()
            machine_id = order.current_location.machine_group
            # Randomly select a row from process_times dataset and retrieve the processing time for the given machine_id
            random_row_index = self.randomStreams["process_time"][machine_id].randint(0, len(process_times))
            step_time = process_times[random_row_index][machine_id]
            total_time += step_time
            order.actual_step += 1  # Advance to next step
        order.actual_step = original_actual_step
        total_time += 1  # Add sink processing time
        return total_time
    #
    def average_remaining_waiting_time(self):
        """
        Calculate average total processing time based on the statistical mean of the process_times dataset.
        """
        # Calculate the average value across all rows in the process_times dataset
        total_sum = 0
        row_count = len(process_times)
        
        for row in process_times:
            total_sum += sum(row)
        
        avg_total = total_sum / row_count
        return avg_total

    def transp_time(self, start, end, transp, statistics, parameters):
        """Return actual processing time for a concrete part."""
        if end.type == "source" or start.id == end.id or start.type == "sink":
            result_time = 0
        elif start.type == "source":
            result_time = parameters['TRANSP_TIME'][start.id][end.machine_group+2]
        elif end.type == "sink":
            result_time = parameters['TRANSP_TIME'][start.machine_group+2][end.id]
        else:
            result_time = parameters['TRANSP_TIME'][start.machine_group+2][end.machine_group+2]
        # statistics['stat_transp_walking'][transp.id] += result_time
        # statistics['stat_transp_working'][transp.id] += result_time
        return result_time

    def handling_time(self, MachineOrSource, LoadOrUnload, transp, statistics, parameters):
        """Return actual handling time for a concrete part."""
        result_time = 0
        if MachineOrSource == "machine":
            if LoadOrUnload == "load":
                result_time += parameters['TIME_TO_LOAD_MACHINE']
            elif LoadOrUnload == "unload":
                result_time += parameters['TIME_TO_UNLOAD_MACHINE']
        elif MachineOrSource == "source":
            if LoadOrUnload == "load":
                result_time += parameters['TIME_TO_LOAD_SOURCE']
            elif LoadOrUnload == "unload":
                result_time += parameters['TIME_TO_UNLOAD_SOURCE'] 
        # statistics['stat_transp_handling'][transp.id] += result_time
        # statistics['stat_transp_working'][transp.id] += result_time
        return result_time

    def changeover_time(self, machine, current_variant, next_variant, statistics, parameters):
        """Return actual changeover time for a machine."""
        result_time = parameters['CHANGEOVER_TIME']
        # statistics['stat_machines_changeover'][machine.machine_group] += result_time
        return result_time

    def time_to_failure(self, machine, statistics, parameters):
        """Return time until next failure for a machine."""
        result_time = self.randomStreams["machine_failure"][machine.machine_group].exponential(scale=parameters['MTBF'][machine.machine_group])
        return result_time

    def repair_time(self, machine, statistics, parameters):
        """Return time until next failure for a machine."""
        result_time = self.randomStreams["repair_time"][machine.machine_group].exponential(scale=parameters['MTOL'][machine.machine_group])
        # statistics['stat_machines_broken'][machine.machine_group] += result_time
        return result_time

    def time_to_order_generation(self, source, statistics, parameters):
        """Return time until next failure for a machine."""
        result_time = self.randomStreams["order_generation"][source.id - self.parmeters['NUM_MACHINES']].exponential(scale=parameters['MTOG'][source.id])
        return result_time
    def create_intermediate_production_steps_and_variant(self, statistics, parameters, resources, at_resource,create_type):
        if create_type == 'P1':
            # result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
            #                     resources['machines'][2], resources['machines'][3], resources['machines'][4],
            #                     resources['machines'][5], resources['machines'][6], resources['sinks'][0]]
            result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
                                resources['machines'][2], resources['machines'][3], resources['machines'][4],
                                resources['machines'][5],resources['machines'][7], resources['machines'][6], resources['sinks'][0]]
        elif create_type == 'P2':
            result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
                                resources['machines'][2], resources['machines'][3], resources['machines'][4],
                                resources['machines'][5], resources['machines'][6], resources['sinks'][0]]
            # result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
            #                     resources['machines'][2], resources['machines'][3], resources['machines'][4],
            #                     resources['machines'][5],resources['machines'][7], resources['machines'][6], resources['sinks'][0]]
        elif create_type == 'P3':
            result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
                                resources['machines'][2], resources['machines'][3], resources['machines'][4],
                                resources['machines'][5], resources['machines'][6], resources['sinks'][0]]
            # result_prod_steps = [at_resource, resources['machines'][0], resources['machines'][1],
            #                     resources['machines'][2], resources['machines'][3], resources['machines'][4],
            #                     resources['machines'][5],resources['machines'][7], resources['machines'][6], resources['sinks'][0]]
        # Sub-component processing steps
        elif create_type == 'P11':  # P11 requires two processing steps
            result_prod_steps = [at_resource,resources['machines'][7], resources['machines'][8],resources['sinks'][0]]
        elif create_type == 'P21':
            result_prod_steps = [at_resource,  resources['machines'][9],resources['sinks'][0]]
        elif create_type == 'P31':  # P31 also requires two processing steps
            result_prod_steps = [at_resource,  resources['machines'][10],resources['machines'][11],resources['sinks'][0]]
        result_variant = self.randomStreams["order_sequence"].choice(parameters['NUM_PROD_VARIANTS'], 1, p=parameters['VARIANT_DISTRIBUTION'])
        return result_prod_steps, result_variant

def update_mov_avg(**kwargs):
    """ Function to iteratively calculate moving average with a given window"""
    kwargs['cont'].appendleft(kwargs['value'])
    new_mean = sum(kwargs['cont']) / len(kwargs['cont'])
    return new_mean

def update_mov_std(**kwargs):
    """ Function to iteratively calculate moving std with a given window"""
    kwargs['cont_sq'].appendleft(kwargs['value'] ** 2)
    len_wd = len(kwargs['cont_sq'])
    new_var = ((1 / len_wd) * sum(kwargs['cont_sq'])) - ((1 / len_wd) * sum(kwargs['cont'])) ** 2

    if new_var < 0:
        new_var = 0
    return np.sqrt(new_var)

def update_exp_weighted_mean(**kwargs):
    """ Function to iteratively calculate exponentially weighted mean"""
    new_mean = (1 - kwargs['alpha']) * kwargs['oldMean'] + kwargs['alpha'] * kwargs['value']
    return new_mean

def update_exp_weightes_std(**kwargs):
    """ Function to iteratively calculate exponentially weighted std"""
    diff = kwargs['value'] - kwargs['oldMean']
    incr = kwargs['alpha'] * diff
    new_var = (1 - kwargs['alpha']) * (kwargs['oldStd'] ** 2 + diff * incr)
    return np.sqrt(new_var)

class ZScoreNormalization(object):
    running_mean = dict(
        exp=update_exp_weighted_mean,
        mov=update_mov_avg
    )
    running_std = dict(
        exp=update_exp_weightes_std,
        mov=update_mov_std
    )

    def __init__(self, type, **config):
        """ Calculates the z score normalization for a given method of calculating the mean and std """
        self.update_mean = ZScoreNormalization.running_mean[type]
        self.update_std = ZScoreNormalization.running_std[type]
        self.config = dict(**config)
        self.type = type
        self.counter = 0
        self.mean = 0
        self.std = 0
        self.setup()

    def __call__(self, value):
        """ By calling this function calculates the mean, std and z score normalization of the next iteration with given value """
        self.counter += 1
        update = dict(
            oldMean=self.mean,
            value=value,
            oldStd=self.std,
            counter=self.counter,
            **self.attr_alg
        )
        self.mean = self.update_mean(**update)
        self.std = self.update_std(**update)

    def setup(self):
        """ Setup function to set all parameters for calculations"""
        if self.type is 'mov':
            size_wd = self.config['window']
            self.attr_alg = dict(cont_sq=deque([], maxlen=size_wd), cont=deque([], maxlen=size_wd), **self.config)
        elif self.type is 'exp':
            self.attr_alg = dict(**self.config)

    def get_z_score_normalization(self, value):
        if self.std != 0 and value != None:
            normalized = (value - self.mean) / self.std
            return normalized
        elif value == None:
            return -1
        else:
            return 0

    def reset(self):
        self.counter = 0
        self.mean = 0
        self.std = 0
        self.setup()
