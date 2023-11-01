import numpy as np
import math


class V2Vchannels:
    # Simulator of the V2V Channels
    '''
    3GPP 37.885 "Study on evaluation methodology of new Vehicle-to-Everything (V2X) use cases for LTE and NR; (Rel. 15)"
    '''
    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 5.9  # GHz
        self.decorrelation_distance = 25
        self.shadow_std = 3
        self.vehAntGain = 3

    def get_path_loss(self, distance, block):
        v2v_distance = abs(distance)*abs(block)
        # if v2v_distance < 1:  # This condition will barely happen. Just to make sure we do not end up in NaN.
        #     v2v_distance = 1

        if block == 1:   # LoS scenario 3GPP 37.885 shadowing
            Path_loss = 38.77 + 16.7 * np.log10(v2v_distance) + 18.2 * np.log10(self.fc)- 2 * self.vehAntGain
        elif block == 0: # self interference
            Path_loss = 100
        else:            # NLoS (We model blockage by adding an additional power reduction)
            Path_loss = 36.85 + 30 * np.log10(v2v_distance) + 18.9 * np.log10(self.fc) + 5*(abs(block))
            # Path_loss = 36.85 + 30 * np.log10(v2v_distance) + 18.9 * np.log10(self.fc)

        return Path_loss

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)


class Vehicle:

    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity


class Environ:

    def __init__(self, width, size_platoon, n_RB, BW, V2V_SIZE, Gap, safety_dist, velocity, N_m, threshold,
                 outage_prob, min_rate, max_power, MATI):
        # Road Configuration
        # 3GPP 37.885 "Study on evaluation methodology of new V2X use cases for LTE and NR". P.37
        up_lanes = [i for i in [4 / 2, 4 / 2 + 4, 4 / 2 + 8]]
        self.lanes = up_lanes
        self.width = width
        self.road_labels = ['lower', 'middle', 'upper']  # line of the highway
        self.velocity = velocity
        self.gap = Gap
        self.safety_dist = safety_dist
        self.n_RB = n_RB
        self.size_platoon = size_platoon
        self.N_Agents = int(self.size_platoon - 2)
        self.bandwidth = BW  # bandwidth per RB, 180,000 MHz
        self.V2V_demand_size = V2V_SIZE  # V2V payload: * Bytes every MATI
        self.Nakagami_m = N_m
        self.threshold_dB = threshold  # dB watt = 10*log10(watt)
        self.threshold = np.power(10, threshold / 10)  # dB watt = 10*log10(watt)
        self.out_prob = outage_prob
        self.min_rate = min_rate
        self.max_power = max_power
        self.MATI_bound = MATI

        self.V2Vchannels = V2Vchannels()
        self.vehicles = []

        self.V2V_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2V_pathloss = []

        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2_ = -174 + 10 * np.log10(BW) + self.vehNoiseFigure
        self.sig2 = 10 ** ((self.sig2_ - 30) / 10)
        self.v_length = 0

        self.V2V_data_rate = np.zeros([self.N_Agents])
        self.platoon_V2V_Interference_db = np.zeros([self.N_Agents])
        self.V2V_powers = np.zeros([self.N_Agents])
        self.V2V_RBs = np.zeros([self.N_Agents])

    def seed(self, seed_val):
        np.random.seed(seed_val)

    def add_new_platoon(self, start_position, start_direction, start_velocity, size_platoon):
        for i in range(size_platoon):
            self.vehicles.append(Vehicle([start_position[0], start_position[1] - i * (self.gap + self.safety_dist)],
                                          start_direction, start_velocity))

    def add_new_platoon_by_number(self, size_platoon, shadowing_dist):

        '''
        it is important to mention that the initial starting points of the platoons
        shall not affect the overall system performance.

        :param shadowing_dist:  disance between the vehicles
        :param size_platoon:    platoon size
        :return:
        '''

        ind = np.random.randint(len(self.lanes))
        start_position = [self.lanes[ind], np.random.randint(450, self.width/10)]  # position of platoon leader
        self.add_new_platoon(start_position, self.road_labels[ind], self.velocity, size_platoon)

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.delta_distance = shadowing_dist

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle within the platoon
        # ===============
        pass

    def renew_channel(self, v2v_dist):
        """ Renew slow fading channel """
        '''
        In calculating the Shadowing for two objects, we calculate the shadowing from the manhattan distance 
        between these two objects. 
        Manhattan Distance Formula:
        In a plane with p1 at (x1, y1) and p2 at (x2, y2), it is |x1 - x2| + |y1 - y2|.
        Shadowing @ time (n) is: S(n) = exp(-D/D_corr).*S(n-1)+sqrt{ (1-exp(-2*D/D_corr))}.*N_S(n)
        D is update distance matrix where D(i,j) is change in distance of link i to j from time n-1 to time n
        '''
        # self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 100 * np.identity(len(self.vehicles))
        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))

        for i in range(len(self.vehicles)):
            for j in range(len(self.vehicles)):
                self.V2V_pathloss[i, j] = self.V2Vchannels.get_path_loss(np.sum(v2v_dist[0]), i-j)

        # self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing
        self.V2V_channels_abs = self.V2V_pathloss[1:, 1:]
        v2v_channels = self.V2V_channels_abs.copy()
        # pathloss matrices
        A_p = v2v_channels[1:, :-1]
        B_p = v2v_channels[~np.eye(v2v_channels.shape[0], dtype=bool)].reshape(v2v_channels.shape[0], -1)[1:, :]
        self.desired_channel_slow = np.power(10, (-A_p / 10)) * np.identity(self.N_Agents)
        self.full_channel_slow = np.power(10, (-B_p / 10))

    def renew_channels_fastfading(self):

        """ Renew fast fading channel """
        # scaling with sqrt(2) is meant to bring down the exponential distribution lambda to one.
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        # self.V2V_channels_with_fastfading = V2V_channels_with_fastfading
        # self.Nakagami_fast_fading = V2V_channels_with_fastfading
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))
        self.Nakagami_fast_fading = V2V_channels_with_fastfading - 10 * np.log10(
            np.random.gamma(self.Nakagami_m, 1/self.Nakagami_m, V2V_channels_with_fastfading.shape))
        # self.V2V_channels_with_fastfading = np.clip(self.V2V_channels_with_fastfading, 70, 110) #Normalization

        ## channel simplified
        v2v_fastfading = self.V2V_channels_with_fastfading.squeeze()
        A_f = v2v_fastfading[1:, :-1]
        B_f = v2v_fastfading[~np.eye(v2v_fastfading.shape[0], dtype=bool)].reshape(v2v_fastfading.shape[0], -1)[1:, :]
        self.desired_channel_fast = np.power(10, (-A_f / 10)) * np.identity(self.N_Agents)
        self.full_channel_fast = np.power(10, (-B_f / 10))

    def Revenue_function(self, quantity, threshold, coeff):
        # SNR outage function definition in the paper
        flag = 0
        if (quantity >= threshold):
            flag = 1

        revenue = coeff * (quantity - threshold)


        return revenue, flag

    def Compute_V2V_data_rate(self, powers, Trx_interval):

        self.V2V_powers = powers.flatten()
        # -------------------------------------------------------------------------
        # fast fading matrices
        # ------------ Compute Interference --------------------
        self.platoon_V2V_Interference = np.zeros([self.N_Agents])  # V2V interferences
        self.platoon_V2V_Signal = np.zeros([self.N_Agents])  # V2V signals
        # Interference calculation
        for i in range(self.N_Agents):
            self.platoon_V2V_Interference[i] = (self.full_channel_fast[i, :] @ powers - powers[i] * self.full_channel_fast[i][i])

        for i in range(self.N_Agents):
            self.platoon_V2V_Signal[i] = powers[i] * self.full_channel_fast[i][i]

        SINR = np.divide(self.platoon_V2V_Signal, (self.platoon_V2V_Interference + self.sig2))

        V2V_Rate = np.log2(1 + SINR)
        self.V2V_data_rate = V2V_Rate.copy()
        self.intraplatoon_rate = V2V_Rate * Trx_interval * self.bandwidth / 1000
        self.platoon_V2V_Interference_db = 10 * np.log10(self.platoon_V2V_Interference.copy())
        self.platoon_V2V_Interference_db[self.platoon_V2V_Interference_db == -math.inf] = -140
        self.platoon_V2V_Interference_db[self.platoon_V2V_Interference_db == math.inf] = -140

        return V2V_Rate, self.intraplatoon_rate, SINR

    def BS_Outage_Reward(self, actions, success_rate, Trx_interval):

        success_rate_ = success_rate.copy()
        per_user_reward = np.zeros(self.N_Agents)
        action_temp = actions.copy()
        V_rate, intra_rate, SINR = self.Compute_V2V_data_rate(action_temp, Trx_interval)
        self.V2V_MATI -= Trx_interval
        SNR_th = self.threshold / (np.log(1 / (1 - self.out_prob)))
        SNR_th = 10 * np.log10(SNR_th)


        for i in range(self.N_Agents):
            snr_rev, snr_flag = self.Revenue_function(10 * np.log10(SINR[i]), SNR_th, 0.5)
            v2v_rev, v2v_flag = self.Revenue_function(V_rate[i], self.min_rate, 0.5)
            if snr_flag:
                self.V2V_demand[i] -= intra_rate[i]
                if self.V2V_demand[i] <= 0:
                    self.V2V_demand[i] = 0
            else:
                snr_rev = snr_rev*7 # 5 previously

            per_user_reward[i] = v2v_rev + snr_rev

        success_rate_[self.V2V_demand <= 0] = 1
        Reward = np.mean(per_user_reward)

        return np.array([Reward]), V_rate, self.V2V_demand, success_rate_, SINR

    def NN_Outage_Reward(self, actions, success_rate, Trx_interval):

        success_rate_ = success_rate.copy()
        per_user_reward = np.zeros(self.N_Agents)
        action_temp = actions.copy()
        V_rate, intra_rate, SINR = self.Compute_V2V_data_rate(action_temp, Trx_interval)
        self.V2V_MATI -= Trx_interval
        SNR_th = self.threshold / (np.log(1 / (1 - self.out_prob)))
        SNR_th = 10 * np.log10(SNR_th)

        for i in range(self.N_Agents):
            snr_rev, snr_flag = self.Revenue_function(10*np.log10(SINR[i]), SNR_th, 0.5)
            v2v_rev, v2v_flag = self.Revenue_function(V_rate[i], self.min_rate, 0.5)
            if snr_flag:
                self.V2V_demand[i] -= intra_rate[i]
                if self.V2V_demand[i] <= 0:
                    self.V2V_demand[i] = 0
            else:
                snr_rev = snr_rev*7 # 5 previously

            per_user_reward[i] = snr_rev + v2v_rev

        success_rate_[self.V2V_demand <= 0] = 1

        return per_user_reward, V_rate, self.V2V_demand, success_rate_, SINR

    def Opt_Outage_Reward(self, actions, success_rate, Trx_interval):

        success_rate_ = success_rate.copy()
        per_user_reward = np.zeros(self.N_Agents)
        action_temp = actions.copy()
        V_rate, intra_rate, SINR = self.Compute_V2V_data_rate(action_temp, Trx_interval)
        self.V2V_MATI -= Trx_interval

        for i in range(self.N_Agents):
            self.V2V_demand[i] -= intra_rate[i]
            if self.V2V_demand[i] <= 0:
                self.V2V_demand[i] = 0
            # per_user_reward[i] = np.tanh(snr_rev) + np.tanh(v2v_rev) - (self.V2V_demand[i] / self.V2V_demand_size)

        success_rate_[self.V2V_demand <= 0] = 1


        return per_user_reward, V_rate, self.V2V_demand, success_rate_, SINR

    def MATI_Reward_design_one(self, control_signals, time_idxs, selected_MATIs, mati_bound, idx):
        '''
        :epsilon: trying to avoid the 0 by 0 division
        :param control_signals: u = KX
        :param idx: ID of the vehicle for which we are trying to compute the string stability.
        :return: The time domain string stability

                                      L2_norm[u_i]
                 string_Stability = ----------------
                                     L2_norm[u_{i-1}]
        '''
        epsilon = 1e-10

        str_stable = (self.integration_mati(control_signals[idx+1], time_idxs) + epsilon) / \
                     (self.integration_mati(control_signals[idx], time_idxs) + epsilon)

        if str_stable <= 1:
            mati_reward = 1 - np.exp(0.1 * str_stable)
        else:
            mati_reward = 1 - np.exp(2 * str_stable) + 6.2839

        if mati_reward <= -47:
            mati_reward = -47
        mati_reward = mati_reward - (5/(selected_MATIs))

        return mati_reward, str_stable

    def MATI_Reward_design_two(self, control_signals, time_idxs, idx):
        '''
        :epsilon: trying to avoid the 0 by 0 division
        :param control_signals: u = KX
        :param idx: ID of the vehicle for which we are trying to compute the string stability.
        :return: The time domain string stability

                                      L2_norm[u_i]
                 string_Stability = ----------------
                                     L2_norm[u_{i-1}]
        '''
        epsilon = 1e-10
        str_stable = (self.integration_mati(control_signals[idx+1], time_idxs)) / \
                     (self.integration_mati(control_signals[0], time_idxs) + epsilon)


        return str_stable

    def integration_mati(self, y, x):

        value = np.trapz(y**2, x)
        value = value ** (1/2)

        return value

    def new_random_game(self, V2V_dist, shadow_dist):

        # make a new game
        self.vehicles = []
        self.add_new_platoon_by_number(self.size_platoon, shadow_dist)
        self.renew_channel(V2V_dist)  # V2V_dist = shadow dist ---> in the beginning
        self.renew_channels_fastfading()
        self.V2V_demand = self.V2V_demand_size * np.ones(int(self.size_platoon-2), dtype=np.float16)
        self.V2V_MATI = self.MATI_bound * np.ones(int(self.size_platoon-2), dtype=int)
