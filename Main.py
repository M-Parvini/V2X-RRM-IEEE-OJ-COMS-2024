'''
The python code for:
            "Bridging the gap between Communication and Control with Artificial
                     Intelligence in platooning vehicular networks"
Written by: Mohammad Parvini, Research Associate at the Technische Universität Dresden

todo: I am not a sophisticated person in naming the variables. Change them if you think they are not obvious.
'''
# imports
import numpy as np
import argparse
import time as scheduling_period
from Networks.TD3_Networks import TD3_ActorNetwork, TD3_CriticNetwork
from RSU import RSU_server
import Classes.Environment_Platoon as ENV
from Classes.Hybrid_System import Hybrid_system
import Utils.utils as OTTO
import os

starting_time = scheduling_period.time()

def create_path(activation, fed_comm, Cent_RL, alg, SINR_th):
    # path creation based on the algorithm selection
    timestr = scheduling_period.strftime("_%M%S")
    if (alg == 'opt_non_fair') or (alg == 'opt_fair') or (alg == 'equal_power'):
        folder = 'results_' + alg + '_' + str(SINR_th)
        save_path = os.path.join(folder, "Saved_data")
    elif fed_comm:
        folder = 'results_' + 'Fed_centralized_' + ('NN' if activation != 'linear' else activation) + '_' + str(SINR_th)
        save_path = os.path.join(folder, "Saved_data")
    elif Cent_RL:
        folder = 'results_' + 'Centralized_' + ('NN' if activation != 'linear' else activation) + '_' + str(SINR_th)
        save_path = os.path.join(folder, "Saved_data")
    else:
        folder = 'results_' + 'decentralized_' + ('NN' if activation != 'linear' else activation) + '_' + str(SINR_th)
        save_path = os.path.join(folder, "Saved_data")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return folder, save_path

def main(platoon_size, road_length, Headway, CACC, Platoon_speed, safety_distance, n_RB, max_power, BW, N_m, SINR_th,
         rate_th, outage_prob, nPointsODEScaling, pert_interval, pert_values, vehPairParams, batch_size, memory_size,
         gamma, actor_lr, critic_lr, epsilon, Critic_dims, Actor_dims, activation, squashing_func, weight_init, taus,
         simulation_time, n_train, n_test, V2V_size, MATI, full_data, algorithm, FCSI, Cent_RL, Fed_Comm, train, test,
         extend, Exp):

    # path creation for saving the data and weights
    folder, save_path = create_path(activation, Fed_Comm, Cent_RL, algorithm, SINR_th)

    # Initialization
    Headway_gap = Platoon_speed / 3.6 * Headway
    initial_velocity = Platoon_speed / 3.6
    n_agents = platoon_size - 2  # excluding the reference vehicle and the fact that the number of links are less by one
    N_veh = platoon_size - 1  # number of vehicles supporting the V2V links
    vehPairParams["kp"] = vehPairParams["kd"] ** 2

    y_initial = np.kron(np.ones([1, platoon_size]), np.array([0, initial_velocity, 0, 0]))
    y_initial = np.block([y_initial, np.zeros([1, N_veh])]).reshape(-1)
    V2V_dist = OTTO.V2V_Dist(N_veh, y_initial.copy(), Headway, safety_distance)

    # envs
    HS_CACC = Hybrid_system(vehPairParams, platoon_size, Headway, CACC, pert_interval, pert_values,
                            nPointsODEScaling, simulation_time, MATI)
    env = ENV.Environ(road_length, platoon_size, n_RB, BW, V2V_size, Headway_gap, safety_distance,
                      initial_velocity, N_m, SINR_th, outage_prob, rate_th, max_power, MATI)
    env.new_random_game(V2V_dist, V2V_dist)  # initialize parameters in env
    # ---------------------------------------------------------------------------------------------------------------- #
    if (algorithm == 'TD3') and (Cent_RL == False): # Semi-distributed machine learning
        n_action = 1  # power control
        outage_input_size = len(OTTO.get_outage_state(env=env, idx=0))
        actor_net = TD3_ActorNetwork(actor_lr, outage_input_size, Actor_dims, activation, squashing_func, weight_init,
                                     n_action)
        critic_net = TD3_CriticNetwork(critic_lr, outage_input_size, Critic_dims, activation, squashing_func, weight_init,
                                       n_action)
    elif (algorithm == 'TD3') and (Cent_RL == True): # Fully centralized machine learning
        n_action = n_agents  # power control
        outage_input_size = len(OTTO.get_total_outage_state(env=env))
        actor_net = TD3_ActorNetwork(actor_lr, outage_input_size, Actor_dims, activation, squashing_func, weight_init,
                                     n_action)
        critic_net = TD3_CriticNetwork(critic_lr, outage_input_size, Critic_dims, activation, squashing_func, weight_init,
                                       n_action)
    elif (algorithm == 'opt_fair') or (algorithm == 'opt_non_fair') or (algorithm == 'equal_power'):
        n_action = None  # RB & Power
        outage_input_size = None
        actor_net = None
        critic_net = None
    else:
        raise ValueError('Wrong Algorithm was chosen')

    RSU = RSU_server(actor_net, critic_net, n_agents, N_veh, env, HS_CACC, epsilon, gamma, taus, n_train, n_test,
                     batch_size, memory_size, Platoon_speed, Headway, safety_distance, simulation_time, full_data,
                     algorithm, Cent_RL, Fed_Comm, save_path, folder, extend, Exp, FCSI)
    if train:
        RSU.train('train')
    if test:
        RSU.test('test')
    if Exp:
        RSU.explainable_AI()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi Agent RL based RRM for CACC enabled Platooning Networks')
    parser.add_argument("--platoon_size", type=int, default=10, help='Platoon Size')
    parser.add_argument("--road_length", type=int, default=5000, help='Highway length')
    parser.add_argument("--Headway", type=float, default=0.5, help='Platoon headway value')
    parser.add_argument("--CACC", type=int, default=1, help='CACC or ACC module definer')
    parser.add_argument("--Platoon_speed", type=int, default=140, help='Platoon cruising speed')
    parser.add_argument("--safety_distance", type=int, default=100, help='inter platoon safety distance [m]')
    parser.add_argument("--n_RB", type=int, default=1, help='number of resource blocks')
    parser.add_argument("--max_power", type=float, default=30, help='vehicles maximum transmit power [dBm]')
    parser.add_argument("--BW", type=int, default=1e6, help='per RB bandwidth [Hz]')
    parser.add_argument("--N_m", type=int, default=1, help='Nakagami fading argument')
    parser.add_argument("--SINR_th", type=float, default=15, help='Minimum required SINR')
    parser.add_argument("--rate_th", type=float, default=5, help='Minimum required data rate [bps/Hz]')
    parser.add_argument("--outage_prob", type=float, default=0.01, help='Outage Probability')
    parser.add_argument("--nPointsODEScaling", type=int, default=20, help='ODE solver points')
    parser.add_argument("--pert_interval", type=list, default=[0, 0.5, 3, 5.5, 8],
                        help='acceleration profile times (in seconds')
    parser.add_argument("--pert_values", type=list, default=[0, -2, 0, 4, 0],
                        help='acceleration profile values (in meters per squared seconds')
    parser.add_argument("--vehPairParams", type=dict, default={"kga": 1, "kgb": 1, "na": 0.1, "nb": 0.1,
                                                               "kd": 0.5}, help='vehicle pair parameters')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--memory_size", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.99, help='RL discount factor')
    parser.add_argument("--actor_lr", type=float, default=0.0001, help='actor nets learning rate')
    parser.add_argument("--critic_lr", type=float, default=0.001, help='critic nets learning rate')
    parser.add_argument("--epsilon", type=int, default=1, help='soft/hard update initial value')
    parser.add_argument("--Critic_dims", type=list, default=[64, 32, 16], help='critic nets hidden layers')
    parser.add_argument("--Actor_dims", type=list, default=[64, 32, 16], help='actor nets hidden layers')
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'leaky_relu', 'elu', 'linear'],
                        help='neural networks activation function')
    parser.add_argument("--squashing_function", action='store_true')
    parser.add_argument("--inverting_gradients", action='store_false')
    parser.add_argument("--weight_init", type=str, default='kaiming',
                        help='neural networks weight initialization type', choices=['kaiming', 'normal', 'phil_tabor'])
    parser.add_argument("--taus", type=list, default=[0.01, 0.001], help='soft update weight')
    parser.add_argument("--simulation_time", type=int, default=30, help='Simulation time')
    parser.add_argument("--n_episodes_train", type=int, default=30, help='total training episodes')
    parser.add_argument("--n_episodes_test", type=int, default=10, help='total test episodes')
    parser.add_argument("--CAM_size", type=float, default=6500 * 8, help='total CAM message payload (bits)')
    parser.add_argument("--MATI", type=int, default=149, help='Maximum Allowable Transmission Interval')
    parser.add_argument("--full_data_loader", type=int, default=20, help="full data to compute the total gradient")
    parser.add_argument("--Algorithm", type=str, default='opt_fair',
                        choices=['TD3', 'opt_non_fair', 'opt_fair', 'equal_power'])
    parser.add_argument("--FCSI", default=True, choices=[True, False]) # Full CSI at BS
    parser.add_argument("--Centralized_RL", default=False, choices=[True, False])  # Full CSI at BS
    parser.add_argument("--federated_communication", default=False, choices=[True, False])
    parser.add_argument("--train", default=False, choices=[True, False])
    parser.add_argument("--test", default=True, choices=[True, False])
    parser.add_argument("--Model_extend", default=False, choices=[True, False])
    parser.add_argument("--explainable_model", default=False, choices=[True, False])
    args = parser.parse_args()

    main(
        platoon_size = args.platoon_size,
        road_length = args.road_length,
        Headway = args.Headway,
        CACC =  args.CACC,
        Platoon_speed =  args.Platoon_speed,
        safety_distance = args.safety_distance,
        n_RB = args.n_RB,
        max_power = args.max_power,
        BW = args.BW,
        N_m = args.N_m,
        SINR_th = args.SINR_th,
        rate_th = args.rate_th,
        outage_prob = args.outage_prob,
        nPointsODEScaling = args.nPointsODEScaling,
        pert_interval = args.pert_interval,
        pert_values = args.pert_values,
        vehPairParams = args.vehPairParams,
        batch_size = args.batch_size,
        memory_size = args.memory_size,
        gamma = args.gamma,
        actor_lr = args.actor_lr,
        critic_lr = args.critic_lr,
        epsilon = args.epsilon,
        Critic_dims = args.Critic_dims,
        Actor_dims = args.Actor_dims,
        activation = args.activation,
        squashing_func = args.squashing_function,
        weight_init = args.weight_init,
        taus = args.taus,
        simulation_time = args.simulation_time,
        n_train = args.n_episodes_train,
        n_test = args.n_episodes_test,
        V2V_size = args.CAM_size,
        MATI = args.MATI,
        full_data = args.full_data_loader,
        algorithm = args.Algorithm,
        FCSI = args.FCSI,
        Cent_RL = args.Centralized_RL,
        Fed_Comm = args.federated_communication,
        train = args.train,
        test = args.test,
        extend = args.Model_extend,
        Exp = args.explainable_model
        )
