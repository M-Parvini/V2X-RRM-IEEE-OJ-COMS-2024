import os
import csv
import matplotlib.pyplot as plt
import torch
import scipy
import numpy as np
from base_funcs.TD3_server import Server
import Utils.utils as OTTO
from V2VAgent import TD3_v2v_agent
from BSAgent import TD3_BS_agent
import shap


class RSU_server(Server):
    def __init__(self, actor_model, critic_model, n_agents, N_veh, env, HS_CACC, epsilon, gamma, taus, n_train, n_test,
    batch_size, mem_size, Platoon_speed, Headway, safety_distance, sim_time, full_data, algorithm, Cent_RL, Fed_Comm,
                 save_path, folder, extend, model_exp, FCSI):
        super().__init__(actor_model, critic_model, n_agents, Cent_RL, Fed_Comm, folder)

        # Initialize
        self.n_state = None if (algorithm == 'opt_fair') or (algorithm == 'opt_non_fair') or (algorithm == 'equal_power') \
            else actor_model.state_size
        self.n_action = None if (algorithm == 'opt_fair') or (algorithm == 'opt_non_fair') or (algorithm == 'equal_power') \
            else actor_model.action_size
        self.n_agents = n_agents
        self.N_veh = N_veh
        self.size_platoon = env.size_platoon
        self.env = env
        self.HS_CACC = HS_CACC
        self.n_train = n_train
        self.n_test = n_test
        self.MATI_quantity = self.HS_CACC.MATI
        self.headway = Headway
        self.Headway_gap = Platoon_speed / 3.6 * Headway
        self.initial_velocity = Platoon_speed / 3.6
        self.safety_distance = safety_distance
        self.sim_time = sim_time
        self.update_cntr = 0
        self.max_power = self.env.max_power
        self.algorithm = algorithm
        self.Fed_Comm = Fed_Comm
        self.Cent_RL = Cent_RL
        self.save_path = save_path
        self.folder = folder
        self.extend = extend
        self.explain = model_exp
        self.FCSI = FCSI
        self.reward_record = []
        self.reward_pre = -100
        # Algorithm initialization
        if (algorithm == 'TD3') and (Cent_RL == False):
            for i in range(self.n_agents):
                id = 'agent_' + str(i)
                user = TD3_v2v_agent(id, gamma, mem_size, epsilon, taus, batch_size, actor_model, critic_model,
                                      full_data, algorithm, Fed_Comm, folder)
                self.users.append(user)
        elif (algorithm == 'TD3') and (Cent_RL == True):
            id = 'BS'
            user = TD3_BS_agent(id, gamma, mem_size, epsilon, taus, batch_size, actor_model, critic_model,
                                 full_data, algorithm, Fed_Comm, folder)
            self.users.append(user)
        elif algorithm == 'opt_fair':
            self.fairness = True

        elif algorithm == 'opt_non_fair':
            self.fairness = False


    def reset(self):
        # Initialization
        self.done = False
        self.time = 0
        self.Trx_Cntr = np.zeros(self.n_agents)
        self.MATIs = np.zeros(self.n_agents, dtype=int) * self.MATI_quantity
        self.success_rate = np.zeros(self.n_agents, dtype=int)
        self.String_Stability = np.zeros(self.n_agents)
        self.acc_data = np.zeros(self.n_agents)
        self.y_initial = np.kron(np.ones([1, self.size_platoon]), np.array([0, self.initial_velocity, 0, 0]))
        self.Total_Output = self.y_initial.copy()
        self.Total_SS = self.String_Stability.copy()
        self.Total_Time = self.time
        self.y_initial = np.block([self.y_initial, np.zeros([1, self.N_veh])]).reshape(-1)
        self.V2V_dist = OTTO.V2V_Dist(self.N_veh, self.y_initial.copy(), self.headway, self.safety_distance)
        self.state_outage_old_all = [0] * self.n_agents
        self.state_outage_new_all = [0] * self.n_agents
        self.env.V2V_demand = np.ones(self.n_agents, dtype=np.float16) * self.env.V2V_demand_size
        self.env.V2V_MATI = self.env.MATI_bound * np.ones(self.n_agents, dtype=int)

    def v2v_passthrough_layer_ini(self):
        # Linear passthrough bias and weight initialization
        initial_weights = np.zeros((self.n_action, self.n_state))
        initial_bias = np.ones(self.n_action) * (-0.8)  # adding a small bias tomake the output comparable to the state
        for user in self.users:
            user.set_passthrough_resenet_weight_biases(initial_weights, initial_bias)

    def save_episodic_results(self, episode):
        total_output_path = os.path.join(self.save_path + '/total_outputs.mat')
        scipy.io.savemat(total_output_path, {'total_outputs': self.Total_Output})

        trx_path = os.path.join(self.save_path + '/trx_' + str(episode) + '_cntr.mat')
        scipy.io.savemat(trx_path, {'trx_' + str(episode) + '_cntr': self.Trx_Cntr})

        total_ss_path = os.path.join(self.save_path + '/total_ss.mat')
        scipy.io.savemat(total_ss_path, {'total_ss': self.Total_SS})

    def save_intermediary_results(self, episode, action_outage, V_rate, outage_reward, SINR, total_state_info, opt):

        with open(self.save_path + '/power_consumption_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(action_outage.flatten())

        with open(self.save_path + '/total_state_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(total_state_info.flatten())

        with open(self.save_path + '/V2V_rate_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(V_rate)

        with open(self.save_path + '/outage_reward_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(outage_reward)

        with open(self.save_path + '/string_stability_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.String_Stability)

        with open(self.save_path + '/SINR_vals_' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SINR)

        with open(self.save_path + '/time' + str(episode) +'_'+ opt+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(np.array([self.time], float))

    def Save_models(self):
        self.save_server_model()
        for user in self.users:
            user.save_agent_model()

    def Optim_max_min_run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound


            # DC programming and finding the optimal power values
            power = OTTO.min_max_outage(self.env, self.FCSI, self.folder)
            action_outage = power

            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Opt_Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            print('Chosen powers: ', action_outage.flatten())
            # print('remaining V2V payload: ', self.env.V2V_demand)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)

    def Optim_DC_programming_run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound


            # DC programming and finding the optimal power values
            power_lb = OTTO.DC_programming(self.env, self.FCSI, self.folder)
            # power = (power_lb+power_ub)/2
            action_outage = power_lb

            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Opt_Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            print('Chosen powers: ', action_outage.flatten())
            # print('remaining V2V payload: ', self.env.V2V_demand)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)


    def Equal_power_run(self, episode, option):

        self.reset()
        self.env.seed(seed_val=np.random.randint(100, 1000))
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage

            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound


            #  Choosing the action for packet transmissions
            # action_outage = np.ones(self.n_agents)*0.5
            action_outage = np.ones(self.n_agents)

            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.Opt_Outage_Reward(action_outage, self.success_rate, time_ - self.time)

            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")
            print('Episode: ', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            print('Chosen powers: ', action_outage.flatten())
            # print('remaining V2V payload: ', self.env.V2V_demand)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)

    def BS_TD3_Run(self, episode, option):

        self.reset()
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds

            #  Getting the initial states
            state_outage_old = OTTO.get_total_outage_state(env=self.env)

            # If the time equals the MATI, then the agent will proceed with a new MATI
            #  In every MATI, a new packet must be transmitted, hence the success rate must be updated
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound

            '''
            In this phase the BS will select agents power and the resources and based on these information,
            the channel gains & data rates of the whole platoon (vehicles) will be determined.
            '''
            action = self.users[0].choose_action(state_outage_old, option)
            action = np.clip(action, 0.001, 0.999)
            # action = np.clip((action+1)/2, 0.001, 0.999)
            #                .
            #  State update: X = AX + Bu
            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.BS_Outage_Reward(action, self.success_rate, time_ - self.time)
            self.reward_record.append(outage_reward)
            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")

            print('Episdoe:', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            state_outage_new = OTTO.get_total_outage_state(env=self.env)

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            # print('Chosen MATIs: ', MATIs)
            print('Chosen powers: ', action.flatten())
            # print('remaining V2V payload: ', env.V2V_demand)
            print('exploration: ', self.users[0].epsilon)
            print('outage performance: ', outage_reward)

            # Save intermediary results
            self.save_intermediary_results(episode, action, V_rate, np.array(outage_reward), SINR, state_outage_old, option)

            # taking the agents actions, states and reward and learning phase
            if option == 'train':
                self.users[0].memory.store_transition(state_outage_old, action,
                                                      outage_reward, state_outage_new, self.done)

                # train
                if self.users[0].memory.mem_cntr >= self.users[0].full_data:  # does not matter which user (same for all)
                    print('============== Training phase ===============')
                    for user in self.users:
                        user.train()
                    if self.Fed_Comm:
                        self.aggregate_parameters()
            # old observation = new_observation
            state_outage_old = state_outage_new

    def Fed_TD3_Run(self, episode, option):

        self.reset()
        #  Renewing the channels --> path-loss + fast fading
        self.env.renew_channel(self.V2V_dist)
        self.env.renew_channels_fastfading()

        while self.time < self.sim_time * 1000:  # changing the time milliseconds
            #  Getting the initial states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_old_all[i] = state_outage
                # print('agent ' + str(i) + ' state', state_outage)

            # If the time equals the MATI, then the agent will proceed with a new MATI
            #  In every MATI, a new packet must be transmitted, hence the success rate must be updated
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    self.Trx_Cntr[i] += 1
                    # Which acceleration data we are transmitting?
                    self.acc_data[i] = self.HS_CACC.current_acceleration_compute(self.MATIs[i])
                    # get the new MATI: agent.get_mati[i]
                    self.MATIs[i] += self.MATI_quantity
                    self.success_rate[i] = 0
                    self.env.V2V_demand[i] = self.env.V2V_demand_size
                    self.env.V2V_MATI[i] = self.env.MATI_bound

            #  Choosing the action for packet transmissions
            action_outage = np.zeros(self.n_agents)  # power, resource block
            action_outage_list = []
            for i in range(self.n_agents):
                '''
                In this phase the agents will select their power and the resources and based on these information,
                the channel gains & data rates of the whole platoon (vehicles) will be determined.
                '''
                action = self.users[i].choose_action(self.state_outage_old_all[i], option)
                action = np.clip(action, 0.001, 0.999)
                action_outage_list.append(action)
                # action = OTTO.Clipping(action)  # No clipping --> gradient inverting
                action_outage[i] = action  # chosen RB
            #                .
            #  State update: X = AX + Bu
            T_new, Y_new = self.HS_CACC.state_update(self.time, self.y_initial.copy(), self.success_rate,
                                                     self.MATIs, self.acc_data)

            y_initial_ = Y_new[-1]
            time_ = int(np.round((T_new[-1])))
            self.Total_Output = np.block([[self.Total_Output], [Y_new[1:, :4 * self.size_platoon]]])
            self.Total_Time = np.block([self.Total_Time, T_new[1:]])
            #  Calculating the reward for the outage and packet transmission part
            outage_reward, V_rate, Demand_R, success_rate_, SINR = \
                self.env.NN_Outage_Reward(action_outage, self.success_rate, time_ - self.time)
            self.reward_record.append(outage_reward)
            #  calculating the new distance by subtracting the distance differences of V2V links
            V2V_dist_ = OTTO.V2V_Dist(self.N_veh, y_initial_.copy(), self.headway, self.safety_distance)
            print("-----------------------------------")

            print('Episdoe:', episode)
            print('Time before:', self.time)
            print('Time after:', time_)
            print('MATIS: ', self.MATIs)
            print('string stability: ', self.String_Stability)

            self.time = time_
            self.y_initial = y_initial_
            self.V2V_dist = V2V_dist_
            self.success_rate = success_rate_

            #  Renewing the channels --> path-loss + fast fading
            self.env.renew_channel(self.V2V_dist)
            self.env.renew_channels_fastfading()

            ## Learning phase for the MATI neural networks
            for i in range(self.n_agents):
                if self.time == self.MATIs[i]:
                    # Some part of the
                    String_, String_time_ = self.HS_CACC.String_Stability_total(self.Total_Output.copy(),
                                                                                self.Total_Time.copy())
                    self.String_Stability[i] = self.env.MATI_Reward_design_two(String_, String_time_, i)

            self.Total_SS = np.block([[self.Total_SS], [self.String_Stability]])

            # Getting the new states
            for i in range(self.n_agents):
                state_outage = OTTO.get_outage_state(env=self.env, idx=i)
                self.state_outage_new_all[i] = state_outage

            #  Changing the done to True in case of terminal state
            if self.time >= self.sim_time * 1000:
                self.done = True

            # print('Chosen MATIs: ', MATIs)
            print('Chosen powers: ', action_outage.flatten())
            # print('remaining V2V payload: ', env.V2V_demand)
            print('exploration: ', self.users[0].epsilon)
            print('outage performance: ', outage_reward)

            # Save intermediary results
            self.save_intermediary_results(episode, action_outage, V_rate, outage_reward, SINR,
                                           np.array(self.state_outage_old_all).flatten(), option)

            # taking the agents actions, states and reward and learning phase
            if option == 'train':
                for i in range(self.n_agents):
                    self.users[i].memory.store_transition(self.state_outage_old_all[i], action_outage_list[i],
                                                          outage_reward[i], self.state_outage_new_all[i], self.done)

                # train
                if self.users[0].memory.mem_cntr >= self.users[0].full_data:  # does not matter which user (same for all)
                    print('============== Training phase ===============')
                    for user in self.users:
                        user.train()
                    if self.Fed_Comm:
                        self.aggregate_parameters()
                # old observation = new_observation
                for i in range(self.n_agents):
                    self.state_outage_old_all[i] = self.state_outage_new_all[i]

    def train(self, training):

        for episode in range(self.n_train):
            print("-------------Round number: ", episode, " -------------")
            if self.Fed_Comm:
                self.send_parameters()
            # self.v2v_passthrough_layer_ini()
            if (self.algorithm == 'TD3') and (self.Cent_RL == False):
                self.Fed_TD3_Run(episode, training)
            elif (self.algorithm == 'TD3') and (self.Cent_RL == True):
                self.BS_TD3_Run(episode, training)
            else:
                raise ValueError("Unknown algorithm " + str(self.algorithm))

            self.save_episodic_results(episode)
            average_reward = np.mean(np.array(self.reward_record))
            if average_reward >= self.reward_pre:
                self.Save_models()
                self.reward_pre = average_reward

    def test(self, testing):
        # if self.extend:
            # self.load_server_model()
            # common_actor, common_critic = self.users[1].load_agent_model(self.server_actor_model, self.server_critic_model, extend=False)

        for user in self.users:
            # user.load_agent_model(self.server_actor_model, self.server_critic_model, self.extend)
            user.load_agent_model()
            user.epsilon = user.eps_min

        for episode in range(self.n_test):

            if (self.algorithm == 'TD3') and (self.Cent_RL == False):
                self.Fed_TD3_Run(episode, testing)

            elif (self.algorithm == 'TD3') and (self.Cent_RL == True):
                self.BS_TD3_Run(episode, testing)

            elif self.algorithm == 'opt_fair':
                self.Optim_max_min_run(episode, 'fair')

            elif self.algorithm == 'opt_non_fair':
                self.Optim_DC_programming_run(episode, 'non_fair')

            elif self.algorithm == 'equal_power':
                self.Equal_power_run(episode, 'equal_power')

            else:
                raise ValueError("Unknown algorithm " + str(self.algorithm))

            self.save_episodic_results(episode)

    def explainable_AI(self):
        #
        i = 0
        # feature_names = ['Path-loss', 'Fast-fading, RB1', 'Fast-fading, RB2', 'Fast-fading, RB3',
        #                  'Interference', 'Remaining time', 'Remaining packet'],
        plt.rcParams["font.family"] = "Times New Roman"
        l = self.users[0].actor_model.state_size

        for user in self.users:
            # user.load_agent_model(self.server_actor_model, self.server_critic_model, self.extend)
            user.load_agent_model(actor_model=None, critic_model=None, extend=self.extend)
            user.epsilon = user.eps_min
            state = np.genfromtxt(os.path.join(self.save_path,'total_state_7_test.csv'), delimiter=',')
            # Choose of number 7 was as a mater of taste. replace it with any number
            state = torch.tensor(state[:,l*i:l*(i+1)]).float().to(user.actor_model.device)
            explainer = shap.DeepExplainer(user.actor_model, state)  # build DeepExplainer
            shap_values = explainer.shap_values(state)  # Calculate shap values
            shap.summary_plot(shap_values, features=state,
                              feature_names=['Fast-fading, RB1', 'Fast-fading, RB2', 'Fast-fading, RB3',
                                             'Interference'],
                              class_names=['RB-power (1)', 'RB-power (2)', 'RB-power (3)'], plot_type = 'bar', show=False)
            plt.legend(frameon=True, framealpha=1, loc='lower right')
            ax = plt.gca()
            ax.grid(b=True, which='major', axis='x', color='#000000', linestyle='--', linewidth=0.25)
            file_name = os.path.join(self.save_path, str(self.folder)+'_'+str(user.id)+'_explainer.pdf')
            plt.savefig(file_name, dpi=500)
            plt.close()
            i += 1
