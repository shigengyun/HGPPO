import datetime
import random
from itertools import chain
import math

from production.envs.time_calc import *
from production.envs.heuristics import *
from production.envs.resources import *
from production.envs.reward_functions import *
import simpy
import numpy as np
from production.envs.logging_config import setup_logging, restore_logging


class Transport(Resource):
    all_transp_orders = []  # Overall list of available transport orders
    agents_waiting_for_action = []
    state_vector = []

    def __init__(self, env, id, resp_area, agent_type, statistics, parameters, resources, agents, time_calc, location,
                 label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Transportation %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.resp_area = resp_area
        self.type = "transp"
        self.idle = env.event()
        self.current_location = self.time_calc.randomStreams["transp_agent"][self.id].choice(
            self.resources["sources"])
        self.transp_log = [["action", "sim_time", "from_at", "to_at", "duration"]]
        self.current_order = None
        self.time_start_idle = 0.0
        self.last_transport_time = 0.0
        self.last_transport_start = 0.0
        self.last_handling_time = 0.0
        self.last_handling_start = 0.0
        self.env.process(self.transporting())
        self.agent_type = agent_type
        self.mapping = None
        if self.agent_type == "FIFO":
            self.agent = Decision_Heuristic_Transp_FIFO(env=self.env, statistics=self.statistics, parameters=self.parameters,
                    resources=self.resources, agents=self.agents, agents_resource=self, time_calc=self.time_calc)
        elif self.agent_type == "EMPTY":
            self.agent = Decision_Heuristic_Transp_EMPTY(env=self.env, statistics=self.statistics, parameters=self.parameters,
                    resources=self.resources, agents=self.agents, agents_resource=self, time_calc=self.time_calc)
        elif self.agent_type == "LIFO":
            self.agent = Decision_Heuristic_Transp_LIFO(env=self.env, statistics=self.statistics, parameters=self.parameters,
                    resources=self.resources, agents=self.agents, agents_resource=self, time_calc=self.time_calc)
        # action_id -> workstation_id mapping (0 = virtual warehouse/source)
        self.mapping = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        self.counter = 0
        self.sum_reward = 0.0
        # GA parameters - lightweight mode by default
        if parameters.get('GA_LIGHTWEIGHT', True):
            self.ga_population_size = 5
            self.ga_generations = 3
            self.ga_mutation_rate = 0.2
            self.ga_crossover_rate = 0.6
            self.ga_elite_size = 1
        else:
            self.ga_population_size = 5
            self.ga_generations = 2
            self.ga_mutation_rate = 0.2
            self.ga_crossover_rate = 0.6
            self.ga_elite_size = 2
        # GA prescreening parameters
        self.ga_selection_ratio = parameters.get('GA_SELECTION_RATIO', 0.8)
        self.min_selected_orders = parameters.get('GA_MIN_SELECTED_ORDERS', 2)
        self.max_selected_orders = parameters.get('GA_MAX_SELECTED_ORDERS', 8)
        self.cache_update_interval = parameters.get('GA_CACHE_UPDATE_INTERVAL', 10)
        self.priority_orders_cache = []
        self.last_cache_update = 0
        self.state_before = self.calculate_state()
        self.next_action = None
        self.latest_reward = 0.0
        self.invalid_counter = 0
        self.next_action_valid = True
        self.next_action_order = None
        self.next_action_origin = None
        self.next_action_destination = None
        self.last_action_id = None
        self.last_reward_calc = 0.0
        self.last_reward_calc_time = 0.0
        self.counter_action_subsets = [0, 0, 0]  # valid, entry, exit
        self.next_destination_problem_order = None
        self.change_tip = False
        self.next_action_reward = 0
        self.time_end_order = 0

    def in_resp_area(self, order):
        return True

    @classmethod
    def put(cls, order, trans_agents):
        if order not in Transport.all_transp_orders:
            Transport.all_transp_orders.append(order)
            for transp_agent in trans_agents:
                if transp_agent.in_resp_area(order):
                    if transp_agent.idle.triggered:
                        idle_time = transp_agent.env.now - transp_agent.time_start_idle
                        transp_agent.transp_log.append(
                            ["idle", round(transp_agent.time_start_idle, 5), transp_agent.current_location.id,
                             transp_agent.current_location.id, round(idle_time, 5)])
                        transp_agent.time_start_idle = 0.0
                        transp_agent.idle = order.env.event()
                        order.env.process(transp_agent.transporting())
                        break

    def get_inventory(self):
        inv = 0
        if self.current_order is not None:
            inv = 1
        return inv

    def discretize_state_value(self, value, thresholds=None):
        # 8-threshold discretization -> 9 bins in [0, 1]
        if thresholds is None:
            thresholds = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95]
        value = max(0.0, min(1.0, value))
        if value < thresholds[0]:
            return 0.0
        elif value < thresholds[1]:
            return 0.125
        elif value < thresholds[2]:
            return 0.25
        elif value < thresholds[3]:
            return 0.375
        elif value < thresholds[4]:
            return 0.5
        elif value < thresholds[5]:
            return 0.625
        elif value < thresholds[6]:
            return 0.75
        elif value < thresholds[7]:
            return 0.875
        else:
            return 1.0

    def calculate_state(self):
        # S1: 13-dim binary availability vector (1 source + 12 machines)
        state_vector = [0.0] * 13
        all_machines = []
        all_machines.extend(self.resources['machines'])
        all_machines.sort(key=lambda x: x.id)
        source = self.resources['sources'][0]
        for order in Transport.all_transp_orders:
            if order.reserved:
                continue
            current_location = order.current_location
            next_step = order.get_next_step()
            can_transport = False
            if next_step.type == "sink":
                can_transport = True
            elif next_step.type == "machine":
                if next_step.is_free():
                    can_transport = True
                elif next_step.is_free_hum():
                    can_transport = True
            if can_transport:
                if current_location.type == "source":
                    state_vector[0] = 1.0
                elif current_location.type == "machine":
                    machine_index = current_location.id
                    state_vector[machine_index - 1] = 1.0

        if 'sys_buffer_state' in self.parameters['TRANSP_AGENT_STATE']:
            # S2: 13-dim station load vector
            station_loads = [0.0] * 13
            source = self.resources['sources'][0]
            source_load = len(source.buffer_out) / max(1, source.capacity)
            source_load_normalized = min(source_load, 1.0)
            station_loads[0] = self.discretize_state_value(source_load_normalized)
            for machine in self.resources['machines']:
                machine_idx = machine.id
                if 1 <= machine_idx <= 12:
                    in_load = len(machine.buffer_in)
                    if machine.buffer_processing:
                        in_load += 1
                    out_load = len(machine.buffer_out)
                    total_load = in_load + out_load
                    total_capacity = machine.capacity * 2 + 1
                    if machine.machine_type == 2:
                        for group_machine in self.resources.get(f'machine_group_{machine.machine_group}', []):
                            group_in_load = len(group_machine.buffer_in)
                            if group_machine.buffer_processing:
                                group_in_load += 1
                            group_out_load = len(group_machine.buffer_out)
                            total_load += group_in_load + group_out_load
                            total_capacity += group_machine.capacity * 2 + 1
                    load_ratio = total_load / max(1, total_capacity)
                    load_ratio_normalized = min(load_ratio, 1.0)
                    station_loads[machine_idx - 1] = self.discretize_state_value(load_ratio_normalized)
            state_vector.extend(station_loads)

        if 'order_state' in self.parameters['TRANSP_AGENT_STATE']:
            # S3: 3-dim global order statistics
            total_waiting_time = 0.0
            total_progress = 0.0
            total_remaining_time = 0.0
            order_count = 0
            for order in Transport.all_transp_orders:
                if not order.reserved:
                    waiting_time = order.get_total_waiting_time()
                    total_waiting_time += waiting_time
                    progress = order.actual_step / max(1, len(order.prod_steps) - 1)
                    total_progress += progress
                    remaining_time = self.time_calc.remaining_steps_processing_time(order)
                    total_remaining_time += remaining_time
                    order_count += 1
            if order_count > 0:
                avg_waiting_time = total_waiting_time / order_count
                normalized_waiting = min(math.log(avg_waiting_time + 1) / math.log(self.time_calc.average_remaining_waiting_time() + 1), 1.0)
                discrete_waiting = self.discretize_state_value(normalized_waiting)
                avg_progress = total_progress / order_count
                discrete_progress = self.discretize_state_value(avg_progress)
                avg_remaining_time = total_remaining_time / order_count
                normalized_remaining = min(avg_remaining_time / self.time_calc.average_remaining_waiting_time(), 1.0)
                discrete_remaining = self.discretize_state_value(normalized_remaining)
            else:
                discrete_waiting = 0.0
                discrete_progress = 0.0
                discrete_remaining = 0.0
            state_vector.extend([discrete_waiting, discrete_progress, discrete_remaining])
        return state_vector

    def ga_prescreen_high_reward_orders(self, all_orders):
        """
        Multi-stage GA prescreening to identify high-reward order subsets.
        """
        if not all_orders:
            return []
        # Stage 1: individual fitness ranking
        order_rewards = []
        for order in all_orders:
            reward = self.calculate_enhanced_order_reward(order)
            order_rewards.append((order, reward))
        order_rewards.sort(key=lambda x: x[1], reverse=True)
        # Stage 2: adaptive selection ratio
        selection_ratio = self.calculate_adaptive_selection_ratio(all_orders, order_rewards)
        target_count = max(self.min_selected_orders,
                          min(self.max_selected_orders,
                              int(len(all_orders) * selection_ratio)))
        # Stage 3: combinatorial GA optimization
        if len(all_orders) > target_count:
            candidate_count = min(len(all_orders), target_count * 2)
            candidates = [item[0] for item in order_rewards[:candidate_count]]
            selected_orders = self.enhanced_ga_selection(candidates, target_count)
        else:
            selected_orders = [item[0] for item in order_rewards]
        # Stage 4: quality validation
        validated_orders = self.validate_selected_orders(selected_orders, all_orders)
        return validated_orders

    def calculate_enhanced_order_reward(self, order):
        """
        Compute per-order fitness for GA prescreening.
        Combines workstation availability, time urgency, and completion complexity.
        """
        reward = 0.0
        next_step = order.get_next_step()
        if next_step:
            if next_step.type == 'sink':
                reward += 150
            elif next_step.type == 'machine':
                current_load = next_step.get_capacity()
                load_ratio = current_load / next_step.capacity
                station_reward = (1 - load_ratio) * 150
                reward += station_reward
        waiting_time = order.get_total_waiting_time()
        avg_waiting = max(1, self.time_calc.average_remaining_waiting_time())
        time_factor = max(0, min((avg_waiting - waiting_time) / avg_waiting, 1))
        time_reward = time_factor * 200
        reward += time_reward
        remaining_steps = len(order.prod_steps) - order.actual_step - 1
        remaining_time = self.time_calc.remaining_steps_processing_time(order)
        if remaining_time > 0:
            complexity_value = max(0, 100 - remaining_time * 0.6 + remaining_steps * 8)
            reward += complexity_value
        return reward

    def calculate_adaptive_selection_ratio(self, all_orders, order_rewards):
        base_ratio = self.ga_selection_ratio
        rewards = [r for _, r in order_rewards]
        if len(rewards) > 1:
            reward_std = np.std(rewards)
            reward_mean = np.mean(rewards)
            cv = reward_std / max(reward_mean, 1)
            if cv > 0.5:
                base_ratio *= 0.8
            elif cv < 0.2:
                base_ratio *= 1.2
        if hasattr(self, 'ga_selection_success_rate'):
            if self.ga_selection_success_rate > 0.65:
                base_ratio *= 0.9
            elif self.ga_selection_success_rate < 0.45:
                base_ratio *= 1.1
        return max(0.3, min(0.8, base_ratio))

    def greedy_topk_selection(self, candidates, target_count):
        """
        PPO+SCPR ablation baseline: state-conditioned greedy top-k ranking.
        Uses the same fitness function as the GA (calculate_enhanced_order_reward)
        but returns top-k tasks by score without any crossover or mutation.
        This isolates the contribution of evolutionary operators vs. adaptive fitness ranking.
        Activated when parameters.get('USE_SCPR_ONLY', False) is True.
        """
        if len(candidates) <= target_count:
            return candidates
        scored = [(order, self.calculate_enhanced_order_reward(order)) for order in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [order for order, _ in scored[:target_count]]

    def enhanced_ga_selection(self, candidates, target_count):
        """
        GA combinatorial selection over candidate orders.
        Uses OX crossover, adaptive mutation, and elitism.
        """
        # SCPR ablation: bypass GA, use greedy top-k ranking only
        if self.parameters.get('USE_SCPR_ONLY', False):
            return self.greedy_topk_selection(candidates, target_count)

        if len(candidates) <= target_count:
            return candidates
        population_size = min(4, len(candidates))
        population = []
        for i in range(population_size):
            if i < population_size // 2:
                individual = list(range(target_count))
            else:
                individual = random.sample(range(len(candidates)), target_count)
            population.append(individual)
        for generation in range(4):
            fitness_scores = []
            for individual in population:
                selected_orders = [candidates[i] for i in individual]
                fitness = self.calculate_enhanced_combination_fitness(selected_orders)
                fitness_scores.append(fitness)
            best_idx = fitness_scores.index(max(fitness_scores))
            elite = population[best_idx][:]
            new_population = [elite]
            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitness_scores, tournament_size=4)
                parent2 = self.tournament_selection(population, fitness_scores, tournament_size=4)
                child1, child2 = self.enhanced_crossover(parent1, parent2, len(candidates))
                child1 = self.adaptive_mutation(child1, len(candidates), target_count, generation)
                child2 = self.adaptive_mutation(child2, len(candidates), target_count, generation)
                new_population.extend([child1, child2])
            population = new_population[:population_size]
        final_fitness = []
        for individual in population:
            selected_orders = [candidates[i] for i in individual]
            fitness = self.calculate_enhanced_combination_fitness(selected_orders)
            final_fitness.append(fitness)
        best_individual = population[final_fitness.index(max(final_fitness))]
        return [candidates[i] for i in best_individual]

    def calculate_enhanced_combination_fitness(self, selected_orders):
        if not selected_orders:
            return 0.0
        total_fitness = 0.0
        individual_rewards = []
        for order in selected_orders:
            reward = self.calculate_enhanced_order_reward(order)
            individual_rewards.append(reward)
            total_fitness += reward
        if individual_rewards:
            avg_reward = np.mean(individual_rewards)
            high_reward_count = sum(1 for r in individual_rewards if r > avg_reward * 1.2)
            concentration_bonus = high_reward_count * 30
            total_fitness += concentration_bonus
        return total_fitness

    def enhanced_crossover(self, parent1, parent2, max_index):
        """Order crossover (OX) for permutation-based individuals."""
        if random.random() > self.ga_crossover_rate:
            return parent1[:], parent2[:]
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child1 = [-1] * size
        child2 = [-1] * size
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        def fill_child(child, other_parent):
            child_set = set(child[start:end])
            fill_pos = list(range(end, size)) + list(range(0, start))
            other_pos = list(range(end, size)) + list(range(0, start))
            for i, pos in enumerate(fill_pos):
                for other_idx in other_pos:
                    if other_parent[other_idx] not in child_set:
                        child[pos] = other_parent[other_idx]
                        child_set.add(other_parent[other_idx])
                        break

        fill_child(child1, parent2)
        fill_child(child2, parent1)
        return child1, child2

    def adaptive_mutation(self, individual, max_index, target_length, generation):
        """Adaptive mutation: rate decreases with generation index."""
        adaptive_rate = self.ga_mutation_rate * (1.0 - generation * 0.1)
        if random.random() > adaptive_rate:
            return individual[:]
        mutated = individual[:]
        mutation_type = random.choice(['swap', 'insert', 'replace'])
        if mutation_type == 'swap' and len(mutated) > 1:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation_type == 'insert':
            if len(mutated) > 0:
                remove_idx = random.randint(0, len(mutated) - 1)
                removed = mutated.pop(remove_idx)
                available = set(range(max_index)) - set(mutated)
                if available:
                    new_idx = random.choice(list(available))
                    mutated.append(new_idx)
                else:
                    mutated.append(removed)
        elif mutation_type == 'replace':
            if len(mutated) > 0:
                replace_pos = random.randint(0, len(mutated) - 1)
                available = set(range(max_index)) - set(mutated)
                if available:
                    mutated[replace_pos] = random.choice(list(available))
        while len(mutated) < target_length:
            available = set(range(max_index)) - set(mutated)
            if available:
                mutated.append(random.choice(list(available)))
            else:
                break
        return mutated[:target_length]

    def validate_selected_orders(self, selected_orders, all_orders):
        """
        Post-selection quality check: if selected mean reward is below 1.5x global mean,
        replace lowest-scoring selected orders with top global orders.
        """
        if not selected_orders:
            return selected_orders
        selected_rewards = [self.calculate_enhanced_order_reward(order) for order in selected_orders]
        all_rewards = [self.calculate_enhanced_order_reward(order) for order in all_orders]
        selected_avg = np.mean(selected_rewards)
        all_avg = np.mean(all_rewards)
        if selected_avg < all_avg * 1.5:
            all_order_rewards = [(order, self.calculate_enhanced_order_reward(order)) for order in all_orders]
            all_order_rewards.sort(key=lambda x: x[1], reverse=True)
            selected_order_rewards = [(order, reward) for order, reward in zip(selected_orders, selected_rewards)]
            selected_order_rewards.sort(key=lambda x: x[1])
            replace_count = min(len(selected_orders) // 3, 2)
            for i in range(replace_count):
                if i < len(all_order_rewards) and i < len(selected_order_rewards):
                    best_order = all_order_rewards[i][0]
                    if best_order not in selected_orders:
                        worst_order = selected_order_rewards[i][0]
                        selected_orders[selected_orders.index(worst_order)] = best_order
        return selected_orders

    def calculate_order_potential_reward(self, order):
        """Compute order potential reward for priority cache scoring."""
        reward = 0.0
        next_step = order.get_next_step()
        if next_step:
            if next_step.type == 'sink':
                reward += 200
            elif next_step.type == 'machine':
                current_load = next_step.get_capacity()
                load_ratio = current_load / max(1, next_step.capacity)
                availability_reward = (1.0 - min(load_ratio, 1.0)) * 70
                reward += availability_reward
        progress = order.actual_step / max(1, len(order.prod_steps) - 1)
        completion_reward = progress * 30
        reward += completion_reward
        waiting_time = order.get_total_waiting_time()
        avg_waiting = max(1, self.time_calc.average_remaining_waiting_time())
        urgency_reward = min((avg_waiting - waiting_time) / avg_waiting * 40, 60)
        urgency_reward = max(0, urgency_reward)
        reward += urgency_reward
        remaining_time = self.time_calc.remaining_steps_processing_time(order)
        if remaining_time > 0:
            time_efficiency_reward = max(0, 30 - remaining_time * 0.4)
            reward += time_efficiency_reward
        return reward

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection: sample tournament_size individuals, return the best."""
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index][:]

    def update_priority_orders_cache(self):
        """Refresh the GA-prescreened high-priority order cache at fixed intervals."""
        if (self.counter - self.last_cache_update) >= self.cache_update_interval:
            self.priority_orders_cache = self.ga_prescreen_high_reward_orders(
                Transport.all_transp_orders
            )
            self.last_cache_update = self.counter

    def get_station_priority_orders(self, station_buffer):
        """Return priority orders in station output buffer, sorted by potential reward."""
        if not station_buffer or not self.priority_orders_cache:
            return []
        priority_orders_in_station = []
        for order in station_buffer:
            if order in self.priority_orders_cache and not order.reserved:
                priority_score = self.calculate_order_potential_reward(order)
                priority_orders_in_station.append((order, priority_score))
        priority_orders_in_station.sort(key=lambda x: x[1], reverse=True)
        return [order for order, score in priority_orders_in_station]

    def ga_rl_integrated_selection(self, available_orders):
        """
        HGPPO integrated selection: GA prescreening followed by RL scoring with epsilon-greedy.
        Falls back to lightweight scoring for simple scenarios.
        """
        if not available_orders:
            return None
        if len(available_orders) == 1:
            return available_orders[0]

        if self.should_use_ga(available_orders):
            high_reward_orders = self.ga_prescreen_high_reward_orders(available_orders)
            candidate_orders = high_reward_orders if high_reward_orders else available_orders
            rl_scores = []
            for order in candidate_orders:
                rl_score = self.calculate_rl_decision_score(order)
                rl_scores.append((order, rl_score))
            rl_scores.sort(key=lambda x: x[1], reverse=True)
            epsilon = 0.1
            if random.random() < epsilon and len(rl_scores) > 1:
                return rl_scores[1][0]
            else:
                return rl_scores[0][0]
        else:
            return self.lightweight_order_selection(available_orders)

    def calculate_rl_decision_score(self, order):
        """RL-aligned scoring combining order fitness, station load, urgency, and completion bonus."""
        score = 0.0
        base_fitness = self.calculate_order_potential_reward(order)
        score += base_fitness * 0.4
        next_step = order.get_next_step()
        if next_step and next_step.type == 'machine':
            current_load = next_step.get_capacity()
            load_ratio = current_load / next_step.capacity
            load_score = (1.0 - min(load_ratio, 1.0)) * 20
            score += load_score
        waiting_time = order.get_total_waiting_time()
        if waiting_time > 0:
            time_urgency = min((1000 - waiting_time) * 0.01, 20)
            score += time_urgency
        if next_step and next_step.type == 'sink':
            score += 60
        return score

    def calculate_reward(self, action):
        result_reward = self.parameters['TRANSP_AGENT_REWARD_INVALID_ACTION']
        result_terminal = False
        if self.invalid_counter < self.parameters['TRANSP_AGENT_MAX_INVALID_ACTIONS']:
            if self.parameters['TRANSP_AGENT_REWARD'] == "valid_action":
                result_reward = get_reward_valid_action(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "utilization":
                result_reward = get_reward_utilization(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "waiting_time_normalized":
                result_reward = get_reward_waiting_time_normalized(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "const_weighted":
                result_reward = get_reward_const_weighted(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "transport_time":
                result_reward = get_reward_transport_time(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "throughput":
                result_reward = get_reward_throughput(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "weighted_objectives":
                result_reward = get_reward_weighted_objectives(self, result_reward)
            elif self.parameters['TRANSP_AGENT_REWARD'] == "conwip":
                result_reward = get_reward_conwip()
            elif self.parameters['TRANSP_AGENT_REWARD'] == "mach_num_cost":
                result_reward = get_reward_mach_cost(self, result_reward)
        else:
            self.invalid_counter = 0
            result_reward = self.parameters['TRANSP_AGENT_REWARD_INVALID_ACTION']

        if self.next_action_valid:
            self.invalid_counter = 0
            self.counter_action_subsets[0] += 1
            if self.next_action_destination != -1 and self.next_action_origin != -1 and self.next_action_destination.type == 'machine':
                self.counter_action_subsets[1] += 1
            elif self.next_action_destination != -1 and self.next_action_origin != -1 and self.next_action_destination.type == 'sink':
                self.counter_action_subsets[2] += 1

        if self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT'] > 0:
            result_reward = 0.0
            if (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'valid' and self.counter_action_subsets[0] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'entry' and self.counter_action_subsets[1] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'exit' and self.counter_action_subsets[2] == self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']) or \
                (self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT_TYPE'] == 'time' and self.env.now - self.last_reward_calc_time > self.parameters['TRANSP_AGENT_REWARD_EPISODE_LIMIT']):
                result_terminal = True
                self.last_reward_calc_time = self.env.now
                self.invalid_counter = 0
                self.counter_action_subsets = [0, 0, 0]
            if result_terminal:
                if self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "utilization":
                    result_reward = get_reward_sparse_utilization(self)
                elif self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "waiting_time":
                    result_reward = get_reward_sparse_waiting_time(self)
                elif self.parameters['TRANSP_AGENT_REWARD_SPARSE'] == "valid_action":
                    result_reward = get_reward_sparse_valid_action(self)
        else:
            self.last_reward_calc_time = self.env.now

        if self.next_action_reward != 0:
            result_reward += self.next_action_reward
            self.next_action_reward = 0
        self.latest_reward = result_reward
        print("reward:", result_reward)
        return result_reward, result_terminal

    def transport_available(self):
        if len(Transport.all_transp_orders) == 0:
            return False
        counter_not_free_source = 0
        for order in Transport.all_transp_orders:
            if self.in_resp_area(order) and not order.reserved:
                if order.get_next_step().is_free():
                    return True
                else:
                    if order.current_location.type == "source":
                        counter_not_free_source += 1
        if counter_not_free_source == len(Transport.all_transp_orders):
            return False

    def get_next_action(self):
        self.counter += 1
        self.parameters['step_criteria'].succeed()
        self.parameters['step_criteria'] = self.env.event()
        Transport.agents_waiting_for_action.append(self)
        yield self.parameters['continue_criteria']

        # Refresh GA prescreened order cache
        if self.parameters.get('USE_GA', False):
            self.update_priority_orders_cache()

        if self.agent_type != "TRPO":
            result_order, result_destination = self.agent.act(Transport.all_transp_orders)
            if result_order and result_destination:
                self.next_action_order = result_order
                result_origin = self.next_action_origin = result_order.current_location
                self.next_action_destination = result_destination
                result_valid = self.next_action_valid = True
                self.next_action[0] = result_destination.id
                step_ratio = self.next_action_order.actual_step / (len(self.next_action_order.prod_steps) - 1)
                step_progress_reward = 200 * math.log1p(9 * step_ratio) / math.log1p(9)
                self.next_action_reward = step_progress_reward
            else:
                result_order = None
                result_origin = None
                result_destination = None
                self.next_action_valid = False
                self.next_action[0] = None
                print("no valid action")
            return result_order, result_destination

        result_order = None
        result_origin = None
        result_dest = None
        result_valid = False
        self.latest_reward = 0.0

        if self.mapping[self.next_action[0]] == 0:
            source = self.resources['sources'][0]
            if source.buffer_out:
                available_orders = []
                for order in source.buffer_out:
                    if not order.reserved:
                        if order.get_next_step().type == 'sink':
                            available_orders.append((order, order.get_next_step()))
                        elif order.get_next_step().is_free():
                            available_orders.append((order, order.get_next_step()))
                        elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                            if self.parameters.get('USE_GA_MACHINE_SELECTION', False):
                                available_machines = self.get_available_machines_for_order(order)
                                if available_machines:
                                    selected_machine = self.ga_machine_selection(order, available_machines)
                                    if selected_machine:
                                        available_orders.append((order, selected_machine))
                            else:
                                for mach in [x for x in chain(self.resources['machines'],
                                                            self.resources.get(f'machine_group_{order.get_next_step().machine_group}', []))
                                            if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                                    available_orders.append((order, mach))
                                    break

                if available_orders:
                    if self.parameters.get('USE_GA', False):
                        orders_only = [order for order, dest in available_orders]
                        selected_order = self.ga_rl_integrated_selection(orders_only)
                        if selected_order:
                            for order, dest in available_orders:
                                if order == selected_order:
                                    result_order = order
                                    result_dest = dest
                                    result_valid = True
                                    break
                    else:
                        selected_order = available_orders[0][0]
                        if selected_order:
                            for order, dest in available_orders:
                                if order == selected_order:
                                    result_order = order
                                    result_dest = dest
                                    result_valid = True
                                    result_order.reserved = True
                                    break
        else:
            machine_id = self.mapping[self.next_action[0]]
            machine = next((m for m in chain(self.resources['machines'], self.resources.get(f'machine_group_{machine_id}', []))
                        if m.id == machine_id), None)

            if machine and machine.buffer_out:
                if self.parameters.get('USE_GA', False):
                    priority_orders = self.get_station_priority_orders(machine.buffer_out)
                    if priority_orders:
                        for priority_order in priority_orders:
                            if not priority_order.reserved:
                                next_step = priority_order.get_next_step()
                                if next_step.type == 'sink':
                                    result_order = priority_order
                                    result_dest = next_step
                                    result_valid = True
                                    result_order.reserved = True
                                    break
                                elif next_step.is_free() and next_step.type == "machine":
                                    result_order = priority_order
                                    result_dest = next_step
                                    result_valid = True
                                    result_order.reserved = True
                                    break
                                elif next_step.is_free_hum() and next_step.type == "machine":
                                    for mach in [x for x in chain(self.resources['machines'],
                                                                self.resources.get(f'machine_group_{next_step.machine_group}', []))
                                               if x.machine_group == next_step.machine_group and x.machine_type == 2]:
                                        priority_order.prod_steps[priority_order.actual_step] = mach
                                        result_order = priority_order
                                        result_dest = mach
                                        result_valid = True
                                        result_order.reserved = True
                                        break
                                    break
                        if result_order:
                            pass
                        else:
                            available_orders = []
                            for order in machine.buffer_out:
                                if not order.reserved:
                                    if order.get_next_step().type == 'sink':
                                        available_orders.append((order, order.get_next_step()))
                                    elif order.get_next_step().is_free():
                                        available_orders.append((order, order.get_next_step()))
                                elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                                    if self.parameters.get('USE_GA_MACHINE_SELECTION', False):
                                        available_machines = self.get_available_machines_for_order(order)
                                        if available_machines:
                                            selected_machine = self.ga_machine_selection(order, available_machines)
                                            if selected_machine:
                                                available_orders.append((order, selected_machine))
                                    else:
                                        for mach in [x for x in chain(self.resources['machines'],
                                                                    self.resources.get(f'machine_group_{order.get_next_step().machine_group}', []))
                                                    if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                                            available_orders.append((order, mach))
                                            break

                            if available_orders:
                                orders_only = [order for order, dest in available_orders]
                                selected_order = self.ga_rl_integrated_selection(orders_only)
                                if selected_order:
                                    for order, dest in available_orders:
                                        if order == selected_order:
                                            result_order = order
                                            result_dest = dest
                                            result_valid = True
                                            result_order.reserved = True
                                            break
                    else:
                        available_orders = []
                        for order in machine.buffer_out:
                            if not order.reserved:
                                if order.get_next_step().type == 'sink':
                                    available_orders.append((order, order.get_next_step()))
                                elif order.get_next_step().is_free():
                                    available_orders.append((order, order.get_next_step()))
                                elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                                    if self.parameters.get('USE_GA_MACHINE_SELECTION', False):
                                        available_machines = self.get_available_machines_for_order(order)
                                        if available_machines:
                                            selected_machine = self.ga_machine_selection(order, available_machines)
                                            if selected_machine:
                                                available_orders.append((order, selected_machine))
                                    else:
                                        for mach in [x for x in chain(self.resources['machines'],
                                                                    self.resources.get(f'machine_group_{order.get_next_step().machine_group}', []))
                                                    if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                                            available_orders.append((order, mach))
                                            break

                        if available_orders:
                            orders_only = [order for order, dest in available_orders]
                            selected_order = self.ga_rl_integrated_selection(orders_only)
                            if selected_order:
                                for order, dest in available_orders:
                                    if order == selected_order:
                                        result_order = order
                                        result_dest = dest
                                        result_valid = True
                                        result_order.reserved = True
                                        break
                else:
                    # No GA: select by longest waiting time (FIFO-like)
                    available_orders = []
                    for order in machine.buffer_out:
                        if not order.reserved:
                            if order.get_next_step().type == 'sink':
                                available_orders.append((order, order.get_next_step()))
                            elif order.get_next_step().is_free():
                                available_orders.append((order, order.get_next_step()))
                            elif order.get_next_step().type == "machine" and order.get_next_step().is_free_hum():
                                if self.parameters.get('USE_GA_MACHINE_SELECTION', False):
                                    available_machines = self.get_available_machines_for_order(order)
                                    if available_machines:
                                        selected_machine = self.ga_machine_selection(order, available_machines)
                                        if selected_machine:
                                            available_orders.append((order, selected_machine))
                                else:
                                    for mach in [x for x in chain(self.resources['machines'],
                                                                self.resources.get(f'machine_group_{order.get_next_step().machine_group}', []))
                                                if x.machine_group == order.get_next_step().machine_group and x.machine_type == 2]:
                                        available_orders.append((order, mach))
                                        break

                    if available_orders:
                        max_waiting_time = -float('inf')
                        selected_order = None
                        for order, dest in available_orders:
                            waiting_time = order.get_total_waiting_time()
                            if waiting_time > max_waiting_time:
                                max_waiting_time = waiting_time
                                selected_order = order
                        if selected_order:
                            for order, dest in available_orders:
                                if order == selected_order:
                                    result_order = order
                                    result_dest = dest
                                    result_valid = True
                                    result_order.reserved = True
                                    break

        # 2% chance of forced disassembly via workstation 5
        if result_order and result_dest:
            if random.random() < 0.02:
                if not self.resources['machines'][5].is_free():
                    for mach in [x for x in chain(self.resources['machines'],
                                                self.resources.get(f'machine_group_{self.resources["machines"][5].machine_group}', []))
                                if x.machine_group == self.resources['machines'][5].machine_group and x.machine_type == 2]:
                        result_order.prod_steps[result_order.actual_step] = mach
                        break
                else:
                    result_order.prod_steps[result_order.actual_step] = self.resources['machines'][5]

        if result_order and result_dest:
            self.next_action_destination = result_dest
            self.next_action_order = result_order
            self.next_action_origin = result_order.current_location
            self.next_action_valid = result_valid
            result_order.prod_steps[result_order.actual_step] = result_dest
            step_ratio = self.next_action_order.actual_step / (len(self.next_action_order.prod_steps) - 1)
            step_progress_reward = 200 * math.log1p(9 * step_ratio) / math.log1p(9)
            self.next_action_reward = step_progress_reward
            Transport.all_transp_orders.pop(Transport.all_transp_orders.index(result_order))
        else:
            print("invalid action")
            self.next_action_destination = None
            self.next_action_order = None
            self.next_action_origin = None
            self.next_action_valid = False
        return result_order, result_dest

    def transporting(self):
        while True:
            order, destination = None, None
            if not self.transport_available():
                self.time_start_idle = self.env.now
                self.idle.succeed()
                break
            order, destination = yield self.env.process(self.get_next_action())
            print("machine_num:", self.parameters['NUM_MACHINES'])
            if order is None:
                self.invalid_counter += 1
                self.next_action_valid = False
                if self.invalid_counter >= self.parameters['TRANSP_AGENT_MAX_INVALID_ACTIONS']:
                    self.transp_log.append(
                        ["invalid_action_limit_forced_waiting", round(self.env.now, 5), self.current_location.id,
                         self.current_location.id, self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION']])
                    yield self.env.timeout(self.parameters['TRANSP_AGENT_WAITING_TIME_ACTION'])
                else:
                    yield self.env.timeout(self.parameters['EPSILON'])
            if order not in [None, -1, -2, -3]:
                # Generate sub-orders at split points
                if order.order_type == "P1" and order.current_location.type == 'machine' and order.proc_steps == 3:
                    order.current_location.sorder_create_machine(order_type='P11')
                elif order.order_type == "P2" and order.current_location.type == 'machine' and order.proc_steps == 4:
                    order.current_location.sorder_create_machine(order_type='P21')
                elif order.order_type == "P3" and order.current_location.type == 'machine' and order.proc_steps == 4:
                    order.current_location.sorder_create_machine(order_type='P31')
                print("order info:", order.id, [x.id for x in order.prod_steps], self.current_location.id,
                      order.current_location.id)
                transp_time = self.time_calc.transp_time(start=self.current_location, end=order.current_location,
                                                         transp=self, statistics=self.statistics,
                                                         parameters=self.parameters)
                self.transp_log.append(
                    ["move_to_empty", round(self.env.now, 5), self.current_location.id, order.current_location.id,
                     round(transp_time, 5)])
                yield self.env.timeout(transp_time)
                self.current_location = order.current_location
                time_start_handling = self.env.now
                self.current_order = order.current_location.get_buffer_out(order)
                order.order_log.append(["picked_up", order.id, round(self.env.now, 5), self.id])
                handling_time = 0.0
                if order.current_location.type == "source":
                    handling_time = self.time_calc.handling_time(MachineOrSource="source", LoadOrUnload="unload",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                elif order.current_location.type == "machine":
                    handling_time = self.time_calc.handling_time(MachineOrSource="machine", LoadOrUnload="unload",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                self.transp_log.append(
                    ["pick_up", round(self.env.now, 5), self.current_location.id, self.current_location.id,
                     round(handling_time, 5)])
                self.last_handling_time = handling_time
                self.last_handling_start = self.env.now
                yield self.env.timeout(handling_time)
                transp_time = self.time_calc.transp_time(start=order.current_location, end=destination, transp=self,
                                                         statistics=self.statistics, parameters=self.parameters)
                self.transp_log.append(["transport", round(self.env.now, 5), order.current_location.id, destination.id,
                                        round(transp_time, 5)])
                self.last_transport_time = transp_time
                self.last_transport_start = self.env.now
                yield self.env.timeout(transp_time)
                self.current_location = destination
                order.order_log.append(["arrived", order.id, round(self.env.now, 5), self.id])
                handling_time = 0.0
                if order.current_location.type == "sink":
                    handling_time = self.time_calc.handling_time(MachineOrSource="source", LoadOrUnload="load",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                elif order.current_location.type == "machine":
                    handling_time = self.time_calc.handling_time(MachineOrSource="machine", LoadOrUnload="load",
                                                                 transp=self, statistics=self.statistics,
                                                                 parameters=self.parameters)
                yield self.env.timeout(handling_time)
                order.time_handling += self.env.now - time_start_handling
                self.transp_log.append(
                    ["put_down", round(self.env.now, 5), self.current_location.id, self.current_location.id,
                     round(handling_time, 5)])
                order.order_log.append(["put_down", order.id, round(self.env.now, 5), self.id])
                destination.put_buffer_in(order)
                order.current_location = destination
                order.proc_steps += 1
                self.current_order = None
                order.transported.succeed()
                order.transported = self.env.event()
                order.reserved = False

    def should_use_ga(self, available_orders):
        return True

    def lightweight_order_selection(self, available_orders):
        """
        Fast multi-criteria scoring as fallback for simple scenarios (avoids full GA overhead).
        """
        if not available_orders:
            return None
        if len(available_orders) == 1:
            return available_orders[0]
        order_scores = []
        for order in available_orders:
            score = 0.0
            next_step = order.get_next_step()
            if next_step.type == 'sink':
                score += 20
            elif next_step.type == 'machine':
                current_load = next_step.get_capacity()
                load_ratio = current_load / next_step.capacity
                score += (1.0 - min(load_ratio, 1.0)) * 10
            waiting_time = order.get_total_waiting_time()
            max_wait = 1000
            wait_score = max(0, min((max_wait - waiting_time) / max_wait, 1.0))
            score += wait_score * 15
            progress = order.actual_step / max(1, len(order.prod_steps) - 1)
            score += progress * 5
            order_scores.append((order, score))
        order_scores.sort(key=lambda x: x[1], reverse=True)
        return order_scores[0][0]

    def ga_machine_selection(self, order, available_machines):
        """
        GA-based workstation selection: minimizes load, penalizes distance.
        """
        if not available_machines or len(available_machines) == 1:
            return available_machines[0] if available_machines else None
        population_size = min(4, len(available_machines))
        generations = 3
        population = []
        for i in range(population_size):
            if i < population_size // 2:
                individual = [min(range(len(available_machines)),
                                key=lambda x: available_machines[x].get_capacity())]
            else:
                individual = [random.randint(0, len(available_machines) - 1)]
            population.append(individual)
        for generation in range(generations):
            fitness_scores = []
            for individual in population:
                machine_idx = individual[0]
                machine = available_machines[machine_idx]
                fitness = self.calculate_machine_fitness(machine, order)
                fitness_scores.append(fitness)
            best_idx = fitness_scores.index(max(fitness_scores))
            elite = population[best_idx][:]
            new_population = [elite]
            while len(new_population) < population_size:
                parent1 = self.tournament_selection_machine(population, fitness_scores)
                parent2 = self.tournament_selection_machine(population, fitness_scores)
                child1, child2 = self.crossover_machine(parent1, parent2, len(available_machines))
                child1 = self.mutate_machine(child1, len(available_machines))
                child2 = self.mutate_machine(child2, len(available_machines))
                new_population.extend([child1, child2])
            population = new_population[:population_size]
        final_fitness = []
        for individual in population:
            machine_idx = individual[0]
            machine = available_machines[machine_idx]
            fitness = self.calculate_machine_fitness(machine, order)
            final_fitness.append(fitness)
        best_individual = population[final_fitness.index(max(final_fitness))]
        return available_machines[best_individual[0]]

    def calculate_machine_fitness(self, machine, order):
        fitness = 0.0
        current_load = machine.get_capacity()
        load_ratio = current_load / machine.capacity
        load_fitness = (1.0 - min(load_ratio, 1.0)) * 100
        fitness += load_fitness
        if hasattr(self, 'current_location') and self.current_location:
            distance_factor = 1.0 / (1.0 + abs(machine.id - self.current_location.id))
            fitness += distance_factor * 20
        if machine.machine_type == 1:
            fitness += 10
        elif machine.machine_type == 2:
            fitness += 5
        if hasattr(machine, 'utilization_rate'):
            fitness += machine.utilization_rate * 15
        return fitness

    def tournament_selection_machine(self, population, fitness_scores, tournament_size=3):
        tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index][:]

    def crossover_machine(self, parent1, parent2, max_machines):
        if random.random() > 0.6:
            return parent1[:], parent2[:]
        child1 = parent2[:]
        child2 = parent1[:]
        return child1, child2

    def mutate_machine(self, individual, max_machines):
        if random.random() > 0.2:
            return individual[:]
        mutated = individual[:]
        mutated[0] = random.randint(0, max_machines - 1)
        return mutated

    def get_available_machines_for_order(self, order):
        """Return all reachable workstations (primary + same-group manual stations) for an order."""
        available_machines = []
        next_step = order.get_next_step()
        if next_step.type == 'sink':
            return [next_step]
        elif next_step.type == 'machine':
            if next_step.is_free():
                available_machines.append(next_step)
            if next_step.is_free_hum():
                for mach in [x for x in chain(self.resources['machines'],
                                            self.resources.get(f'machine_group_{next_step.machine_group}', []))
                            if x.machine_group == next_step.machine_group and x.machine_type == 2]:
                    if mach.is_free():
                        available_machines.append(mach)
        return available_machines
