from itertools import chain
import math

from production.envs.order import Order
from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
import simpy
import numpy as np
from production.envs.logging_config import setup_logging, restore_logging


class Machine(Resource):
    agent = None
    counter_order = 70000  # sub-order id counter

    def __init__(self, env, id, capacity, agent_type, resp_area, machine_group, machine_type, statistics, parameters, resources, agents,
                 time_calc, location, label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Machine %s created" % id)
        self.env = env
        self.destroy = False
        self.id = id
        self.label = label
        self.capacity = capacity
        self.broken = False
        self.last_repair_time = 0.0
        self.type = "machine"
        self.current_mounted_variant = 0
        self.time_broken_left = 0.0
        self.idle = env.event()
        self.last_broken_start = 0.0
        self.last_broken_time = 0.0
        self.last_process_start = 0.0
        self.last_process_start_stat = 0.0
        self.last_process_time = 0.0
        self.last_process_end = 0.0
        self.buffer_in = []
        self.buffer_processing = None
        self.buffer_out = []
        self.machine_log = [["action", "sim_time", "at_ID", "duration"]]
        self.time_start_idle = 0.0
        self.time_start_idle_stat = 0.0
        self.machine_idel_time = 0.0
        self.process = self.env.process(self.processing())

        self.agent_type = agent_type
        if agent_type == "FIFO":
            self.agent = Decision_Heuristic_Machine_FIFO(env=self.env, statistics=statistics, parameters=parameters,
                                                         resources=resources, agents=agents, agents_resource=self, time_calc=time_calc)
        self.resp_area = resp_area
        self.machine_group = machine_group
        self.machine_type = machine_type

        if self.machine_type == 1:
            self.env.process(self.break_machine())
        self.counter = 0
        self.sum_reward = 0
        self.last_utilization_calc = self.env.now
        self.track_in_use = []
        self.track_failures = []
        self.machine_wt_normalizer = None
        self.last_machine_cost_calc = self.env.now

    def is_free(self):
        currently_processing = 0
        if self.buffer_processing is not None:
            currently_processing = 1
        return len(self.buffer_in) + len(self.buffer_out) + currently_processing <= 8

    def get_capacity(self):
        currently_processing = 0
        if self.buffer_processing is not None:
            currently_processing = 1
        return len(self.buffer_in) + len(self.buffer_out) + currently_processing

    def is_free_hum(self):
        for mach in [x for x in chain(self.resources['machines'], self.resources.get(f'machine_group_{self.machine_group}', []))
                                  if x.machine_group == self.machine_group and x.machine_type == 2]:
            if mach.is_free():
                return True
        return False

    def is_free_machine_group(self):
        for mach in [x for x in self.resources['machines'] if self.machine_group == x.machine_group]:
            if mach.is_free():
                return True
        return False

    def get_max_waiting_time(self):
        max_wt = None
        if len(self.buffer_out) > 0:
            max_wt = max([order.get_total_waiting_time() for order in self.buffer_out])
        return max_wt

    def get_normalized_wt_all_machines(self):
        return self.machine_wt_normalizer.get_z_score_normalization(self.get_max_waiting_time())

    def get_inventory(self):
        currently_processing = 0
        if self.buffer_processing is not None:
            currently_processing = 1
        return len(self.buffer_in) + len(self.buffer_out) + currently_processing

    def get_cost(self):
        """Compute workstation energy cost over the last scheduling step."""
        start = self.last_machine_cost_calc
        end = self.env.now
        time_period = end - start
        working_time = 0.0
        failure_time = 0.0
        for interval in self.track_in_use:
            if start < interval[1] and interval[0] < end:
                working_time += min(end, interval[1]) - max(start, interval[0])
            elif interval[1] <= start:
                self.track_in_use.remove(interval)
        if self.buffer_processing is not None and not self.broken:
            working_time += end - max(self.last_process_start_stat, start)
        for interval in self.track_failures:
            if start < interval[1] and interval[0] < end:
                failure_time += min(end, interval[1]) - max(start, interval[0])
            elif interval[1] <= start:
                self.track_failures.remove(interval)
        if self.broken:
            failure_time += end - max(self.last_broken_start, start)
        self.last_machine_cost_calc = self.env.now
        if time_period - failure_time == 0.0:
            return 0.0
        idel_time = time_period - failure_time - working_time
        if idel_time < 0:
            return 0.0
        current_load = self.get_inventory()
        load_factor = 1.0 + math.log(1.0 + 15 * current_load / self.capacity)
        if self.machine_type == 1:
            result_cost = working_time * self.parameters['MACHINE_COST'][0] + idel_time * self.parameters['MACHINE_COST'][1]
        else:
            result_cost = working_time * self.parameters['MACHINE_COST'][2] + idel_time * self.parameters['MACHINE_COST'][3]
        result_cost *= load_factor
        return result_cost

    def get_utilization_step(self):
        """Compute machine utilization over the last scheduling step: working_time / (period - failure_time)."""
        result_utilization = 0.0
        start = self.last_utilization_calc
        end = self.env.now
        time_period = end - start
        working_time = 0.0
        failure_time = 0.0
        for interval in self.track_in_use:
            if start < interval[1] and interval[0] < end:
                working_time += min(end, interval[1]) - max(start, interval[0])
            elif interval[1] <= start:
                self.track_in_use.remove(interval)
        if self.buffer_processing is not None and not self.broken:
            working_time += end - max(self.last_process_start_stat, start)
        for interval in self.track_failures:
            if start < interval[1] and interval[0] < end:
                failure_time += min(end, interval[1]) - max(start, interval[0])
            elif interval[1] <= start:
                self.track_failures.remove(interval)
        if self.broken:
            failure_time += end - max(self.last_broken_start, start)
        self.last_utilization_calc = self.env.now
        if time_period - failure_time == 0.0:
            return 1.0
        result_utilization = working_time / (time_period - failure_time)
        if result_utilization > 1.0 + self.parameters['EPSILON'] or result_utilization < 0.0 - self.parameters['EPSILON']:
            print(working_time, time_period)
            print(result_utilization)
            raise Exception("Step utilization infeasible!")
        return result_utilization

    def reactivate_transport_if_idle(self):
        """Wake up any idle transport agent so it can pick up newly available orders."""
        for transp_agent in self.resources['transps']:
            if transp_agent.idle.triggered:
                idle_time = transp_agent.env.now - transp_agent.time_start_idle
                transp_agent.transp_log.append(
                    ["idle", round(transp_agent.time_start_idle, 5), transp_agent.current_location.id,
                     transp_agent.current_location.id, round(idle_time, 5)])
                self.machine_idel_time += idle_time
                transp_agent.time_start_idle = 0.0
                transp_agent.idle = self.env.event()
                self.env.process(transp_agent.transporting())

    def put_buffer_in(self, order):
        if not self.is_free: raise Exception('Machine is not free / no capacity!')
        self.buffer_in.append(order)
        if self.idle.triggered:
            if self.broken:
                idle_time = 0.0
            else:
                idle_time = self.env.now - self.time_start_idle
            self.machine_idel_time += idle_time
            self.machine_log.append(["idle_starvation", round(self.time_start_idle, 5), self.id, round(idle_time, 5)])
            self.time_start_idle = 0.0
            self.idle = self.env.event()
            self.process = self.env.process(self.processing())

    def get_buffer_in(self, order):
        self.reactivate_transport_if_idle()
        if order is None:
            return self.buffer_in.pop(0)
        else:
            return self.buffer_in.pop(self.buffer_in.index(order))

    def get_buffer_in_by_index(self, id):
        if id is None:
            return None
        else:
            return self.buffer_in.pop(id)

    def put_buffer_out(self, order):
        self.buffer_out.append(order)
        self.reactivate_transport_if_idle()

    def get_buffer_out(self, order):
        self.machine_wt_normalizer(order.get_total_waiting_time())
        result_order = self.buffer_out.pop(self.buffer_out.index(order))
        return result_order

    def get_next_action(self):
        self.counter += 1
        result_order = None
        reward = None
        state_before = self.calculate_state()
        action = self.agent.act(states=state_before)
        result_order = self.get_buffer_in(action[0])
        if self.parameters['PRINT_CONSOLE']: print("MACHINE-Action: ActionID ", action[0].id, " - Reward: ", reward)
        return result_order

    def calculate_state(self):
        if self.agent_type == "FIFO":
            return self.buffer_in

    def sorder_create_machine(self, order_type):
        print("generating sub-part disassembly steps")
        prod_steps, variant = self.time_calc.create_intermediate_production_steps_and_variant(
            statistics=self.statistics, parameters=self.parameters, resources=self.resources, at_resource=self,
            create_type=order_type)
        order = Order(env=self.env, id=Machine.counter_order, prod_steps=prod_steps, variant=variant,
                      statistics=self.statistics, parameters=self.parameters, resources=self.resources,
                      agents=self.agents, time_calc=self.time_calc, order_type=order_type)
        Machine.counter_order += 1
        order.set_sop()
        order.current_location = self
        self.put_buffer_out(order)
        order.order_log.append(["sorder_created", order.id, round(self.env.now, 5), self.id])
        self.env.process(order.order_processing())

    def processing(self):
        """Main processing loop: dequeues orders from buffer_in and processes them."""
        while True:
            if len(self.buffer_in) == 0:
                self.time_start_idle = self.env.now
                self.time_start_idle_stat = self.env.now
                self.idle.succeed()
                break

            if self.broken:
                try:
                    with self.resources['repairman'].request(priority=1) as req:
                        yield req
                        yield self.env.timeout(max(self.parameters['EPSILON'], self.time_broken_left - self.env.now))
                        self.time_broken_left = self.last_repair_time = 0.0
                    if self.parameters['PRINT_CONSOLE']: print("Machine %s repaired" % self.id)
                    self.broken = False
                    continue
                except simpy.Interrupt:
                    if self.destroy:
                        return
                    continue

            order = self.get_next_action()
            if order is not None and self.destroy != True:
                self.buffer_processing = order
                time_processing = 0.0
                if self.current_mounted_variant != order.get_next_step():
                    time_processing += self.time_calc.changeover_time(machine=self,
                                                                      current_variant=self.current_mounted_variant,
                                                                      next_variant=order.get_next_step(),
                                                                      statistics=self.statistics,
                                                                      parameters=self.parameters)
                    self.machine_log.append(["changeover", round(self.env.now, 5), self.id, str([self.current_mounted_variant, "->", order.get_next_step()])])
                    self.current_mounted_variant = order.get_next_step()
                time_processing += self.time_calc.processing_time(parameters=self.parameters, order=order)
                # Manual stations have reduced effective processing time
                if self.machine_type == 2:
                    time_processing *= 0.30
                time_processing *= 0.25
                order.time_processing += time_processing
                self.machine_log.append(["processing", round(self.env.now, 5), self.id, round(time_processing, 5)])
                order.order_log.append(["start_processing", order.id, round(self.env.now, 5), self.id])
                self.last_process_start = self.env.now
                self.last_process_time = time_processing
                while time_processing:
                    try:
                        start_time = self.env.now
                        self.last_process_start_stat = self.env.now
                        yield self.env.timeout(time_processing)
                        time_processing = 0
                    except simpy.Interrupt:
                        if self.destroy:
                            self.broken = True
                            break
                        self.track_in_use.append([start_time, self.env.now])
                        order.order_log.append(
                            ["interrupt_processing_start", order.id, round(self.env.now, 5), self.id])
                        self.broken = True
                        self.last_broken_start = self.env.now
                        self.last_broken_time = self.last_repair_time
                        time_processing -= self.env.now - start_time
                        if not self.destroy:
                            with self.resources['repairman'].request(priority=1) as req:
                                yield req
                                print("Machine", self.id, "is repaired in", self.last_repair_time)
                                self.machine_log.append(
                                    ["breakdown", round(self.env.now, 5), self.id, round(self.last_repair_time, 5)])
                                start_time = self.env.now
                                yield self.env.timeout(self.last_repair_time)
                                self.track_failures.append([start_time, self.env.now])
                        order.order_log.append(["interrupt_processing_end", order.id, round(self.env.now, 5), self.id])
                        self.time_broken_left = 0.0
                        self.last_repair_time = 0.0
                        self.broken = False
                self.track_in_use.append([start_time, self.env.now])
                order.order_log.append(["end_processing", order.id, round(self.env.now, 5), self.id])
                self.last_process_end = self.env.now
                self.buffer_processing = None
                order.order_log.append(["put_outbound_buffer", order.id, round(self.env.now, 5), self.id])
                print("machine id:", self.id, "order completed:", order.id, "processing time:", order.time_processing)
                # Back-pressure: wait if output buffer is full
                while len(self.buffer_out) > self.capacity:
                    try:
                        yield self.env.timeout(self.parameters['EPSILON'])
                    except simpy.Interrupt:
                        if self.destroy:
                            return
                        continue
                self.put_buffer_out(order)
                order.remaining_steps -= 1
                order.processed.succeed()

    def kill_process(self, destroy=False):
        if self.process is not None and self.process.is_alive:
            if destroy:
                self.process.interrupt()
                destroy = False

    def break_machine(self):
        """Periodic random failure process (automatic stations only)."""
        while not self.destroy:
            try:
                time_to_next_failure = self.time_calc.time_to_failure(machine=self, statistics=self.statistics,
                                                                      parameters=self.parameters)
                yield self.env.timeout(time_to_next_failure)
            except simpy.Interrupt:
                if self.destroy:
                    return
                continue

            if not self.broken and self.destroy != True:
                if self.parameters['PRINT_CONSOLE']: print("Machine %s is broken" % self.id)
                self.last_repair_time = self.time_calc.repair_time(machine=self, statistics=self.statistics,
                                                                   parameters=self.parameters)
                self.time_broken_left = self.env.now + self.last_repair_time
                if self.idle.triggered:
                    self.broken = True
                    self.last_broken_start = self.env.now
                    self.last_broken_time = self.last_repair_time
                    start_time = self.env.now
                    idle_time = self.env.now - self.time_start_idle
                    self.machine_idel_time = idle_time
                    self.machine_log.append(
                        ["idle_starvation", round(self.time_start_idle, 5), self.id, round(idle_time, 5)])
                    self.time_start_idle = 0.0
                    self.machine_log.append(["breakdown", round(self.env.now, 5), self.id,
                                             round((self.time_broken_left - self.env.now), 5)])
                    try:
                        yield self.env.timeout(self.last_repair_time)
                    except simpy.Interrupt:
                        if self.destroy:
                            return
                        continue
                    self.track_failures.append([start_time, self.env.now])
                    self.broken = False
                    if len(self.buffer_in) == 0:
                        self.time_start_idle = self.env.now
                        self.time_start_idle_stat = self.env.now
                else:
                    if self.process and self.process.is_alive:
                        self.process.interrupt()


def other_jobs(env, repairman):
    """Repairman background job (low priority, preempted by machine repairs)."""
    while True:
        done_in = 1000.0
        while done_in:
            with repairman.request(priority=2) as req:
                yield req
                try:
                    start_time = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start_time
