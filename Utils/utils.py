import numpy as np
import csv
import os
import torch
from Utils import soft_update
import cvxpy as cp


def Clipping(val):

    clipped = np.clip(val, -0.999, 0.999)
    return clipped


def get_outage_state(env, idx):
    """ Get states related to the outage from the environment """
    """
    Normalization:
    large scale fading is around -95dB to -70dB; for that we run the environment and looked at the histogram plot.
    Small scale fading is around -110dB to -70dB; --> histogram plot
    """

    mean_val = -132.76
    std_val = 25.58

    # V2V_abs = (-env.desired_channel_slow[idx, idx] - MIN)/(Rng)

    V2V_fast = (10*np.log10(env.full_channel_fast[idx, idx])-mean_val)/std_val
    # V2V_fast = (-10*np.log10(env.full_channel_fast[idx, idx]))/(Rng)

    # Remaining_mati = env.V2V_MATI[idx]/env.MATI_bound

    # V2V_load_remaining = np.asarray([env.V2V_demand[idx] / env.V2V_demand_size])

    V2V_interference = np.asarray([(env.platoon_V2V_Interference_db[idx]-mean_val)/std_val])
    # V2V_interference = np.asarray([(env.platoon_V2V_Interference_db[idx])/(2*Rng)])

    # return np.concatenate((np.reshape(V2V_fast, -1), V2V_interference, V2V_load_remaining,
    #                        np.reshape(Remaining_mati, -1)), axis=0)
    return np.concatenate((np.reshape(V2V_fast, -1), V2V_interference), axis=0)

def get_total_outage_state(env):
    """ Get states related to the outage from the environment """
    """
    Normalization:
    The program was run and the data set of the channel gains and interference was collected. We found the mean and 
    STD of the data set and rescale the data set as
    Y = (X - E[X]) / std(X)
    """

    mean_val = -132.76
    std_val = 25.58

    # V2V_abs = (-env.desired_channel_slow[idx, idx] - MIN)/(Rng)

    # V2V_fast = (-10*np.log10(np.diag(env.full_channel_fast)))/(Rng)
    V2V_fast = (10*np.log10(env.full_channel_fast.flatten())-mean_val)/std_val

    # Remaining_mati = env.V2V_MATI[idx]/env.MATI_bound

    # V2V_load_remaining = np.asarray([env.V2V_demand/ env.V2V_demand_size])
    V2V_load_remaining = env.V2V_demand/ env.V2V_demand_size

    # V2V_interference = np.asarray([(env.platoon_V2V_Interference_db)/(2*Rng)])
    # V2V_interference = -env.platoon_V2V_Interference_db/(2*Rng)

    # return np.concatenate((np.reshape(V2V_fast, -1), V2V_interference, V2V_load_remaining), axis=0)
    # return np.concatenate((np.reshape(V2V_fast, -1), V2V_interference), axis=0)
    return V2V_fast

def V2V_Dist(N_veh, y_ini, headwayVals, d_r0):

    Ini_V2V_dist = y_ini[4:-N_veh]  #
    Ini_V2V_dist = Ini_V2V_dist[np.array(range(N_veh)) * 4 + 1] * headwayVals + d_r0
    distance = Ini_V2V_dist  # is the change in distance of link i to j (necessary to calculate the shadowing, D_ij)
    return distance

def create_path(objs, directory):
    paths = []
    for i in range(len(objs)):
        paths.append(os.mkdir(os.path.join(directory, "model/" + objs[i])))

    return paths

def action_wrapper(element):
    return 10*np.log10((element/1e-3))

def action_unwrapper(act, max_power):
    '''
    :param act: NN output value
    :param max_power: maximum power
    :return: scaled version of the action between 1 and 30
    '''
    return np.clip(((act + 1) / 2) * max_power, 0, max_power)
    # return np.round(np.clip(((act + 1) / 2) * max_power, 0, max_power))

def FD_merge(nets, n_agents, taus, omega=0.5):
    # only one task == outage::0
    # full update :: Also known as Federated Averaging
    # actor
    with torch.no_grad():
        omega_ = (1-omega)/(n_agents-1)
        for agent_no in range(n_agents):
            agent_state_dict = dict(nets[agent_no][0].actor.named_parameters())
            for name in agent_state_dict:
                agent_state_dict[name] = omega * agent_state_dict[name].clone()
                for idx in range(n_agents):
                    idx_state_vals = dict(nets[idx][0].actor.named_parameters())
                    if agent_no != idx:
                        agent_state_dict[name] += omega_ * idx_state_vals[name].clone()
            soft_update(nets[agent_no][0].actor, nets[agent_no][0].target_actor, taus[1])

    # critic
    with torch.no_grad():
        omega_ = (1-omega)/(n_agents-1)
        for agent_no in range(n_agents):
            agent_state_dict = dict(nets[agent_no][0].critic.named_parameters())
            for name in agent_state_dict:
                agent_state_dict[name] = omega * agent_state_dict[name].clone()
                for idx in range(n_agents):
                    idx_state_vals = dict(nets[idx][0].critic.named_parameters())
                    if agent_no != idx:
                        agent_state_dict[name] += omega_ * idx_state_vals[name].clone()
            soft_update(nets[agent_no][0].critic, nets[agent_no][0].target_critic, taus[0])

def update_network_parameters(self, tau=None):
    actor_params = self.actor.named_parameters()
    critic_params = self.critic.named_parameters()
    target_actor_params = self.target_actor.named_parameters()
    target_critic_params = self.target_critic.named_parameters()

    critic_state_dict = dict(critic_params)
    actor_state_dict = dict(actor_params)
    target_critic_state_dict = dict(target_critic_params)
    target_actor_state_dict = dict(target_actor_params)

    for name in critic_state_dict:
        critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                  (1 - tau) * target_critic_state_dict[name].clone()

    for name in actor_state_dict:
        actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                 (1 - tau) * target_actor_state_dict[name].clone()

    self.target_critic.load_state_dict(critic_state_dict)
    self.target_actor.load_state_dict(actor_state_dict)
    # self.target_critic.load_state_dict(critic_state_dict, strict=False)
    # self.target_actor.load_state_dict(actor_state_dict, strict=False)

def DC_programming(Env, FCSI, folder):

    '''
    Difference of convex functions method for solving the sum_rate maximization,
    max: sum rate
    s.t. C1: outage probabilities
         C2: power constraints
    '''
    BW = Env.bandwidth
    min_rate = Env.min_rate
    n_links = Env.N_Agents

    # powers
    max_power = Env.max_power
    initial_val = 1
    p_max = 10 ** ((max_power - 30) / 10)  # scalar val
    P_max = np.ones(n_links) * p_max
    P_min = np.zeros(n_links)
    P_ini = np.ones(n_links) * initial_val
    P_ini_mat = np.tile(P_ini, [n_links, 1])

    # coefficients
    gamma0 = Env.threshold
    p0 = Env.out_prob
    beta_lb = p0 / (1 - p0)
    beta_lb_ = beta_lb + gamma0
    beta_ub = np.log(1/(1-p0))
    beta_ub_ = beta_ub + gamma0
    epsi = 1e-3
    error = 1  # initial error value for the while loop
    scale = 1e12
    # pathloss matrices
    A_p = Env.desired_channel_slow * scale
    B_p = Env.full_channel_slow * scale

    # fast fading matrices
    A_f = Env.desired_channel_fast * scale
    B_f = Env.full_channel_fast * scale

    if not FCSI:
        A_f = A_p.copy()
        B_f = B_p.copy()
    # Noise
    sig2 = Env.sig2 * scale
    sig2_vec = np.ones(n_links) * sig2

    ## Start of optimization for lower bound problem

    itr = 0
    while abs(error) >= epsi:
        # DC programming main loop
        itr += 1
        P = cp.Variable(shape=n_links, pos=True)
        gradient_val = gradient_cal(A_f, B_f, P_ini_mat, n_links, sig2).flatten()
        f = cp.sum(cp.log(sig2_vec + B_f @ P)/np.log(2))
        g = np.sum(np.log2(sig2_vec + (B_f-A_f) @ P_ini)) + (gradient_val) @ (P - P_ini)

        # Objective
        objective = cp.Maximize(f-g)
        constraints = [P >= P_min, P <= P_max, (gamma0*B_p - A_p * beta_lb_) @ P <= 0,
                       A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]

        # constraints
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break
        r_k = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        r_kk = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        error = abs(r_k - r_kk)
        if itr == 50:
            break

        problem.status
    P_lb = P_ini
    #######################

    '''
    ## For seeing the convergence behavior
    itr_tot = 30
    rate_iter = np.zeros([2, itr_tot])
    rate_iter[0,0] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
    for i in range(itr_tot-1):
        # DC programming main loop
        P = cp.Variable(shape=n_links, pos=True)
        gradient_val = gradient_cal(A_f, B_f, P_ini_mat, n_links, sig2).flatten()
        f = cp.sum(cp.log(sig2_vec + B_f @ P)/np.log(2))
        g = np.sum(np.log2(sig2_vec + (B_f-A_f) @ P_ini)) + (gradient_val) @ (P - P_ini)

        # Objective
        objective = cp.Maximize(f-g)
        constraints = [P >= P_min, P <= P_max, (gamma0*B_p - A_p * beta_lb_) @ P <= 0,
                       A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]

        # constraints
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break

        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        r_kk = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
        rate_iter[0, i+1] = r_kk

    P_ini = np.ones(n_links) * initial_val
    P_ini_mat = np.tile(P_ini, [n_links, 1])
    rate_iter[1, 0] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
    for i in range(itr_tot-1):
        # DC programming main loop
        P = cp.Variable(shape=n_links, pos=True)
        gradient_val = gradient_cal(A_f, B_f, P_ini_mat, n_links, sig2).flatten()
        f = cp.sum(cp.log(sig2_vec + B_f @ P)/np.log(2))
        g = np.sum(np.log2(sig2_vec + (B_f-A_f) @ P_ini)) + (gradient_val) @ (P - P_ini)

        # Objective
        objective = cp.Maximize(f-g)
        constraints = [P >= P_min, P <= P_max, (gamma0*B_p - A_p * beta_ub_) @ P <= 0,
                       A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]

        # constraints
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break

        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        r_kk = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
        rate_iter[1, i+1] = r_kk
    '''
    ## Start of optimization for upper bound problem
    '''
    itr = 0
    while abs(error) >= epsi:
        # DC programming main loop
        itr += 1
        P = cp.Variable(shape=n_links, pos=True)
        gradient_val = gradient_cal(A_f, B_f, P_ini_mat, n_links, sig2).flatten()
        f = cp.sum(cp.log(sig2_vec + B_f @ P) / np.log(2))
        g = np.sum(np.log2(sig2_vec + (B_f - A_f) @ P_ini)) + (gradient_val) @ (P - P_ini)
        objective = cp.Maximize(f - g)
        constraints = [P >= P_min, P <= P_max, (gamma0 * B_p - A_p * beta_ub_) @ P <= 0]
        # constraints = [P >= P_min, P <= P_max]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=True)

        r_k = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        r_kk = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        error = abs(r_k - r_kk)

        if itr == 50:
            break
    
    # P_ub = P_ini
    '''
    # np.savetxt("sumrate_p1.csv", rate_iter, delimiter=",")

    return P_lb

def Energy_efficiency(Env, FCSI):

    '''
    Difference of convex functions method for solving the sum_rate maximization,
    max: sum rate
    s.t. C1: outage probabilities
         C2: power constraints
    '''
    BW = Env.bandwidth
    min_rate = Env.min_rate
    n_links = Env.N_Agents

    # powers
    max_power = Env.max_power
    p_max = 10 ** ((max_power - 30) / 10)  # scalar val
    P_max = np.ones(n_links) * p_max
    P_min = np.zeros(n_links)
    P_ini = np.ones(n_links) * p_max / 2
    P_ini_mat = np.tile(P_ini, [n_links, 1])

    # coefficients
    gamma0 = Env.threshold
    p0 = Env.out_prob
    beta_lb = p0 / (1 - p0)
    beta_lb_ = beta_lb + gamma0
    beta_ub = np.log(1/(1-p0))
    beta_ub_ = beta_ub + gamma0
    epsi = 1e-3
    error = 1  # initial error value for the while loop
    scale = 1e10
    # pathloss matrices
    A_p = Env.desired_channel_slow * scale
    B_p = Env.full_channel_slow * scale

    # fast fading matrices
    A_f = Env.desired_channel_fast * scale
    B_f = Env.full_channel_fast * scale

    if not FCSI:
        A_f = A_p.copy()
        B_f = B_p.copy()
    # Noise
    sig2 = Env.sig2 * scale
    sig2_vec = np.ones(n_links) * sig2

    ## Start of optimization for lower bound problem
    # Rate_0 = Rate_cal(P_ini, B_f, sig2, BW, n_links)
    itr = 0
    while abs(error) >= epsi:
        # DC programming main loop
        itr += 1
        P = cp.Variable(shape=n_links, pos=True)
        gradient_val = gradient_cal(A_f, B_f, P_ini_mat, n_links, sig2).flatten()
        f = cp.sum(cp.log(sig2_vec + B_f @ P)/np.log(2))
        g = np.sum(np.log2(sig2_vec + (B_f-A_f) @ P_ini)) + (gradient_val) @ (P - P_ini)

        # Objective
        objective = cp.Minimize(cp.sum(P))
        constraints = [P >= P_min, P <= P_max, (gamma0*B_p - A_p * beta_lb_) @ P <= 0,
                       A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]
        # constraints
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break
        r_k = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        r_kk = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

        error = abs(r_k - r_kk)
        sum_power = np.sum(P_ini)
        if itr == 50:
            break

    P_lb = P_ini
    return P_lb


def min_max_outage(Env, FCSI, folder):

    '''
    Minimizing the maximum outage as a reliability factor,
    max: min rate
    s.t. C1: power boundaries
         C2: outage constraint
         C3. power boundaries

    to solve this we change the optimization problem into
    max eta
    s.t. C1: rate >= eta
         C2: outage constraint
         C3. power boundaries
    '''
    BW = Env.bandwidth
    min_rate = Env.min_rate
    n_links = Env.N_Agents
    initial_val = 0.5

    # powers
    max_power = Env.max_power
    p_max = 10 ** ((max_power - 30) / 10)  # scalar val
    P_max = np.ones(n_links) * p_max
    P_min = np.zeros(n_links)
    P_ini = np.ones(n_links) * initial_val
    P_ini_mat = np.tile(P_ini, [n_links, 1])

    # coefficients
    gamma0 = Env.threshold
    p0 = Env.out_prob
    beta_lb = p0 / (1 - p0)
    beta_lb_ = beta_lb + gamma0
    beta_ub = np.log(1/(1-p0))
    beta_ub_ = beta_ub + gamma0
    epsi = 1e-3
    error = 1  # initial error value for the while loop
    # B_f_scale = Env.full_channel_fast
    # min_val = np.min(B_f_scale)
    # max_val = np.max(B_f_scale)
    # scale = np.power(10, -(np.log10(min_val)+np.log10(max_val))/2)
    scale = 1e13
    # pathloss matrices
    A_p = Env.desired_channel_slow * scale
    B_p = Env.full_channel_slow * scale

    # fast fading matrices
    A_f = Env.desired_channel_fast * scale
    B_f = Env.full_channel_fast * scale

    if not FCSI:
        A_f = A_p.copy()
        B_f = B_p.copy()
    # Noise
    sig2 = Env.sig2 * scale
    sig2_vec = np.ones(n_links) * sig2

    ## Start of optimization for lower bound problem
    Rate_0 = Rate_cal(P_ini, B_f, sig2, BW, n_links)

    itr = 0
    while abs(error) >= epsi:
        # DC programming main loop
        itr += 1
        P = cp.Variable(shape=n_links, pos=True)
        eta = cp.Variable(shape=1, pos=True)
        f = cp.log(sig2_vec + B_f @ P)/np.log(2)
        g = np.log(sig2_vec + (B_f-A_f) @ P_ini)/np.log(2)
        grad = []
        for idx in range(n_links):
            grad.append(gradient_cal_per_link(A_f, B_f, P_ini_mat, n_links, sig2, idx).flatten()@(P - P_ini))

        objective = cp.Maximize(eta)
        constraints = [P >= P_min, P <= P_max, (gamma0 * B_p - A_p * beta_ub_) @ P <= 0,
                       f - g - cp.hstack(grad) >= eta, A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]
        # constraints = [P >= P_min, P <= P_max]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break
        # Initial value of rate
        Rate_k = Rate_cal(P_ini, B_f, sig2, BW, n_links)
        r_k = np.min(Rate_k)

        # Replacing the optimum value of power
        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        # Next value of rate
        Rate_kk = Rate_cal(P_ini, B_f, sig2, BW, n_links)
        r_kk = np.min(Rate_kk)
        error = abs(r_kk-r_k)
        sum_power = np.sum(P_ini)
        sum_rate = np.sum(Rate_kk)
        if itr == 100:
            break

    # convergence analysis
    '''
    itr_tot = 30
    rate_iter = np.zeros([2, itr_tot])
    rate_iter[0, 0] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
    for i in range(itr_tot-1):
        # DC programming main loop
        P = cp.Variable(shape=n_links, pos=True)
        eta = cp.Variable(shape=1, pos=True)
        f = cp.log(sig2_vec + B_f @ P)/np.log(2)
        g = np.log(sig2_vec + (B_f-A_f) @ P_ini)/np.log(2)
        grad = []
        for idx in range(n_links):
            grad.append(gradient_cal_per_link(A_f, B_f, P_ini_mat, n_links, sig2, idx).flatten()@(P - P_ini))

        objective = cp.Maximize(eta)
        constraints = [P >= P_min, P <= P_max, (gamma0 * B_p - A_p * beta_lb_) @ P <= 0,
                       f - g - cp.hstack(grad) >= eta, A_f@P + (1-np.power(2, min_rate))*(sig2_vec + (B_f-A_f) @ P)>=0]
        # constraints = [P >= P_min, P <= P_max]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break

        # Replacing the optimum value of power
        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        rate_iter[0, i+1] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))

    P_ini = np.ones(n_links) * initial_val
    P_ini_mat = np.tile(P_ini, [n_links, 1])
    rate_iter[1, 0] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
    for i in range(itr_tot-1):
        # DC programming main loop
        P = cp.Variable(shape=n_links, pos=True)
        eta = cp.Variable(shape=1, pos=True)
        f = cp.log(sig2_vec + B_f @ P) / np.log(2)
        g = np.log(sig2_vec + (B_f - A_f) @ P_ini) / np.log(2)
        grad = []
        for idx in range(n_links):
            grad.append(gradient_cal_per_link(A_f, B_f, P_ini_mat, n_links, sig2, idx).flatten() @ (P - P_ini))

        objective = cp.Maximize(eta)
        constraints = [P >= P_min, P <= P_max, (gamma0 * B_p - A_p * beta_ub_) @ P <= 0,
                       f - g - cp.hstack(grad) >= eta,
                       A_f @ P + (1 - np.power(2, min_rate)) * (sig2_vec + (B_f - A_f) @ P) >= 0]
        # constraints = [P >= P_min, P <= P_max]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        if problem.status == 'infeasible':
            break

        # Replacing the optimum value of power
        P_ini = P.value
        P_ini_mat = np.tile(P_ini, [n_links, 1])

        rate_iter[1, i + 1] = np.sum(Rate_cal(P_ini, B_f, sig2, BW, n_links))
    '''
    # np.savetxt("maxmin_p1.csv", rate_iter, delimiter=",")

    return P_ini

def gradient_cal(A_f_, B_f_, P_ini, N, sig2):
    # Iterative gradient calculation
    B_f = B_f_*P_ini
    np.fill_diagonal(B_f, 0)
    grad_val = np.zeros(N)
    ISI_per_link = sig2+np.sum(B_f, 1)
    for i in range(N):
        grad_val += (1/ISI_per_link[i])*(B_f_[i, :]-A_f_[i, :])

    return 1/(np.log(2))*grad_val

def gradient_cal_per_link(A_f_, B_f_, P_ini, N, sig2, idx):
    # Iterative gradient calculation
    B_f = B_f_*P_ini
    np.fill_diagonal(B_f, 0)
    grad_val = np.zeros(N)
    ISI_per_link = sig2+np.sum(B_f, 1)
    grad_val = (1/ISI_per_link[idx])*(B_f_[idx, :]-A_f_[idx, :])

    return 1/(np.log(2))*grad_val

def Rate_cal(P_ini, B_f, sig2, B, N):
    SINR = np.zeros(N)
    Rate = np.zeros(N)
    for i in range(N):
        SINR[i] = (P_ini[i]*B_f[i][i])/(B_f[i,:]@P_ini - P_ini[i]*B_f[i][i] + sig2)
        Rate[i] = np.log2(1+SINR[i])

    return Rate

def Intermediate_matrix(B_p, P_ini, N, sig2, gamma0):
    # Iterative gradient calculation
    Mat = np.zeros([N, N])
    for i in range(N):
        for k in range(N):
            if i==k:
                Mat[i, k] = 0
            else:
                Mat[i, k] = (P_ini[i]/P_ini[k])*np.log2(1+(gamma0*B_p[i,k]*P_ini[k])/(B_p[i,i]*P_ini[i]))

    return Mat
