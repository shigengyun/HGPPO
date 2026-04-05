from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.order import *
import simpy


class Sink(Resource):
    buffer_in = []

    def __init__(self, env, id, statistics, parameters, resources, agents, time_calc, location, label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Sink %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.type = "sink"
        self.buffer_in_indiv = []
        self.order_process_time = 0
        self.sub_order_process_time = 0
        self.order_processed_count = 0
        self.sub_order_processed_count = 0
        self.order_wait_time = 0
        self.sub_order_wait_time = 0
        self.last_order_count = 0
        self.sub_last_order_count = 0
        self.last_order_process_time = 0
        self.sub_last_order_process_time = 0
        self.last_order_wait_time = 0
        self.sub_last_order_wait_time = 0
        self.total_order_process_time = 0

    def put_buffer_in(self, order):
        self.buffer_in_indiv.append(order)
        Sink.buffer_in.append(order)
        order.order_log.append(["sink", order.id, round(self.env.now, 5), self.id])
        self.order_process_time += order.time_processing
        self.order_processed_count += 1
        self.order_wait_time += order.get_total_waiting_time()
        if len(Sink.buffer_in) >= self.parameters['NUM_ORDERS'] - 1:
            print("All orders processed")
            self.parameters['stop_criteria'].succeed()

    def get_order_reward(self):
        order_count = self.order_processed_count - self.last_order_count
        order_wait_time = self.order_wait_time - self.last_order_wait_time
        if self.order_processed_count != 0 and order_count != 0:
            order_time = (self.order_process_time - self.last_order_process_time) / order_count
        else:
            order_time_reward = 0
        self.last_order_count = self.order_processed_count
        self.last_order_process_time = self.order_process_time
        self.last_order_wait_time = self.order_wait_time
        if order_count != 0 and len(self.buffer_in) > 0:
            order = self.buffer_in[-1]
            base_reward = 1 / (1 + (order_wait_time / len(order.prod_steps)))
            if order.order_type in ['P1', 'P2', 'P3']:
                reward = base_reward * 1.3
            else:
                reward = base_reward * 1.1
        else:
            reward = 0
        return reward

    def is_free(self):
        return True

    def is_free_machine_group(self):
        return True
