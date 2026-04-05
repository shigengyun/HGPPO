import random
from itertools import chain

from production.envs.resources import Resource
from production.envs.time_calc import *


"""
Heuristic Decision Agents
"""


class Decision_Heuristic(object):
    flag_order = None

    def __init__(self, env, statistics, parameters, resources, agents, agents_resource, time_calc, location):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location=None)
        self.statistics = statistics
        self.parameters = parameters
        self.resources = resources
        self.agents = agents
        self.env = env
        self.agents_resource = agents_resource
        agents.update({'Decision_Heuristic_Transp': []})
        agents.update({'Decision_Heuristic_Machine': []})

    def act(self, states):
        raise NotImplementedError

    def get_next_machine_hum(self, order, statistics, parameters, resources):
        """Return next manual machine with smallest buffer fill in same group."""
        result_machine = None
        min_buffer_fill = 5
        if order.get_next_step().machine_type == 1:
            for mach in [x for x in resources['machines'] if x.machine_type == 2]:
                if len(mach.buffer_in) < min_buffer_fill:
                    result_machine = mach
                    min_buffer_fill = len(mach.buffer_in)
        return result_machine, min_buffer_fill

    def get_next_machine_min_buffer_fill(self, order, statistics, parameters, resources):
        """Return next machine with smallest relative buffer fill in same group."""
        result_machine = order.get_next_step()
        min_buffer_fill = len(result_machine.buffer_in) / (9 - len(result_machine.buffer_in) - len(result_machine.buffer_out))
        return result_machine, min_buffer_fill


class Decision_Heuristic_Transp_EMPTY(Decision_Heuristic):
    """Select order whose destination workstation has minimum load."""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource, time_calc):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource, time_calc=time_calc, location=None)
        agents['Decision_Heuristic_Transp'].append(self)
        print("EMPTY_Transp_Decision created")

    def act(self, states):
        if states is None:
            return None, None
        result_order = None
        result_dest = None
        min_fill_level = float('inf')
        for order in states:
            if order.reserved:
                continue
            if order.get_next_step().is_free():
                dest = order.get_next_step()
                if dest.type == "sink":
                    result_order = order
                    result_dest = dest
                    break
                else:
                    dest_fill_level = len(dest.buffer_in) + len(dest.buffer_out)
                    if dest_fill_level < min_fill_level:
                        min_fill_level = dest_fill_level
                        result_order = order
                        result_dest = dest
        if result_order and result_dest:
            result_order = states.pop(states.index(result_order))
            result_order.reserved = True
            result_order.prod_steps[result_order.actual_step] = result_dest
        else:
            result_order = None
            result_dest = None
        return result_order, result_dest


class Decision_Heuristic_Transp_FIFO(Decision_Heuristic):
    """Select next transport order by longest total waiting time (FIFO proxy)."""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource, time_calc):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource, time_calc=time_calc, location=None)
        agents['Decision_Heuristic_Transp'].append(self)
        print("FIFO_Transp_Decision created")

    def act(self, states):
        if states is None:
            return None, None
        result_order = None
        result_dest = None
        for order in sorted(states, key=lambda x: x.get_total_waiting_time()):
            if order.get_next_step().is_free() and not order.reserved:
                result_dest = order.get_next_step()
                result_order = order
                break
            elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum() and not order.reserved:
                for mach in [x for x in chain(self.resources['machines'],
                                            self.resources.get(f'machine_group_{order.get_next_step().machine_group}', []))
                            if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                    result_dest = mach
                    result_order = order
                    break
        if result_order and result_dest:
            # 2% chance of forced disassembly via workstation 5
            if random.random() < 0.02:
                if not self.resources['machines'][5].is_free():
                    for mach in [x for x in chain(self.resources['machines'],
                                                self.resources.get(f'machine_group_{self.resources["machines"][5].machine_group}', []))
                                if x.machine_group == self.resources['machines'][5].machine_group and x.machine_type == 2]:
                        break
                else:
                    result_order.prod_steps[result_order.actual_step] = self.resources['machines'][5]
            result_order = states.pop(states.index(result_order))
            result_order.reserved = True
            result_order.prod_steps[result_order.actual_step] = result_dest
        else:
            result_order = None
            result_dest = None
        return result_order, result_dest


class Decision_Heuristic_Transp_LIFO(Decision_Heuristic):
    """Select next transport order by longest remaining processing time (LIFO proxy)."""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource, time_calc):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource, time_calc=time_calc, location=None)
        agents['Decision_Heuristic_Transp'].append(self)
        self.time_calc = time_calc
        print("LIFO_Transp_Decision created")

    def act(self, states):
        if states is None:
            return None, None
        result_order = None
        result_dest = None
        for order in sorted(states, key=lambda x: self.time_calc.remaining_steps_processing_time(x), reverse=True):
            if order.get_next_step().is_free() and not order.reserved:
                result_dest = order.get_next_step()
                result_order = order
                break
        if result_order and result_dest:
            if random.random() < 0.02:
                if not self.resources['machines'][5].is_free():
                    for mach in [x for x in chain(self.resources['machines'],
                                                self.resources.get(f'machine_group_{self.resources["machines"][5].machine_group}', []))
                                if x.machine_group == self.resources['machines'][5].machine_group and x.machine_type == 2]:
                        break
                else:
                    result_order.prod_steps[result_order.actual_step] = self.resources['machines'][5]
            result_order = states.pop(states.index(result_order))
            result_order.reserved = True
            result_order.prod_steps[result_order.actual_step] = result_dest
        else:
            result_order = None
            result_dest = None
        return result_order, result_dest


class Decision_Heuristic_Machine_FIFO(Decision_Heuristic):
    """Machine agent: select next order to process (FIFO)."""
    def __init__(self, env, statistics, parameters, resources, agents, agents_resource, time_calc):
        super(self.__class__, self).__init__(env=env, statistics=statistics, parameters=parameters, resources=resources, agents=agents, agents_resource=agents_resource, time_calc=time_calc, location=None)
        agents['Decision_Heuristic_Machine'].append(self)
        print("FIFO_Machine_Decision created")

    def act(self, states):
        if states is None:
            return None
        for order in states:
            return [order]
