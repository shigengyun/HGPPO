from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.transport import *
import simpy

from production.envs.transport import Transport


class Order(Resource):
    """An order specifies a production request.
    An order has an *id* and a sequence of *prod_steps* to fulfill.
    """

    def __init__(self, env, id, prod_steps, variant, statistics, parameters, resources, agents, time_calc, order_type):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, None)
        self.env = env
        self.id = id
        self.prod_steps = prod_steps
        self.variant = variant
        self.sop = -1
        self.eop = -1
        self.time_processing = 0
        self.time_handling = 0
        self.actual_step = 0
        self.finished = False
        self.current_location = None
        self.order_log = [["action", "order_ID", "sim_time", "resource_ID"]]
        self.transported = self.env.event()
        self.processed = self.env.event()
        self.reserved = False
        self.time_end_processed = False
        if self.parameters['PRINT_CONSOLE']: print("Order %s created %s" % (self.id, [x.id for x in self.prod_steps]))
        self.order_log.append(["prod_steps", self.id, [x.id for x in prod_steps]])
        self.order_type = order_type
        self.rcount = 0
        self.proc_steps = 0
        self.remaining_steps = len(self.prod_steps) - 2  # source and sink excluded

    def set_sop(self):
        self.sop = self.env.now
        self.order_log.append(["sop", self.id, round(self.sop, 5), ""])

    def set_eop(self):
        self.eop = self.env.now
        self.order_log.append(["eop", self.id, round(self.eop, 5), ""])

    def set_next_step(self):
        self.actual_step += 1
        if self.actual_step > len(self.prod_steps):
            self.finished = True

    def get_next_step(self):
        return self.prod_steps[self.actual_step]

    def get_total_waiting_time(self):
        result = self.env.now - self.sop - self.time_processing - self.time_handling
        return result

    def order_processing(self):
        while True:
            self.set_next_step()
            if self.finished:
                break
            if self.id >= 0 and self in self.current_location.buffer_out:
                self.order_log.append(["before_transport", self.id, round(self.env.now, 5), self.current_location.id])
                Transport.put(order=self, trans_agents=self.resources['transps'])
                yield self.transported
                self.transported = self.env.event()
                self.order_log.append(["after_transport", self.id, round(self.env.now, 5), self.current_location.id])
            if self.get_next_step().type == 'sink':
                break
            self.order_log.append(["before_processing", self.id, round(self.env.now, 5), self.current_location.id])
            yield self.processed
            self.processed = self.env.event()
            self.order_log.append(["after_processing", self.id, round(self.env.now, 5), self.current_location.id])
        self.set_eop()
        self.current_location = None
