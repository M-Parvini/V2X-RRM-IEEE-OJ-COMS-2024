import numpy as np
from Classes.Platoon import Platoon
from Classes.Platoon_ODE import ode_platoon
from scipy.integrate import solve_ivp

np.random.seed(1997)

class Hybrid_system:
    def __init__(self, vehPairParams, N_v, Headway, v, pert_intervals, pert_values, nPointsODEScaling, sim_time, mati):

        self.Platoon = Platoon(vehPairParams, N_v, Headway, v)
        self.Platoon.Create_platoon_Mats()
        self.pert_intervals = np.array(pert_intervals)*1000
        self.pert_values = np.array(pert_values)
        self.ODEScaling = nPointsODEScaling
        self.n_V2V_links = N_v-2
        self.size_platoon = N_v
        self.sim_time = sim_time*1000
        self.MATI = mati
        '''
        defining the Big matrices to calculate the differential equations:
        dX(t)/dt = A11.X(t) + A12.eu(t) + A13.u0(t)
        deu(t)/dt = A21.X(t) + A22.eu(t) + A23.u0(t)
        '''
        self.A11 = self.Platoon.A + np.dot((self.Platoon.Bs + self.Platoon.Bc), self.Platoon.K)
        self.A12 = self.Platoon.Bc
        self.A13 = self.Platoon.Br
        self.A21 = (-1)*np.dot(self.Platoon.K, self.A11)
        self.A22 = (-1)*np.dot(self.Platoon.K, self.A12)
        self.A23 = (-1)*np.dot(self.Platoon.K, self.A13)
        self.AA = np.block([[self.A11, self.A12], [self.A21, self.A22]])
        self.BB = np.block([[self.A13], [self.A23]])
        self.seqPacketloss = 0
        self.seqPacketlossMax = 3

    def state_update(self, time, y_ini, success_rate, MATIs, acc_vals):
        """
        :param t_sim: current simulation time, starting point for the ODE Solver
        :param y_ini: initial values of the system
        :param success_rate: parameter showing whether the acceleration packet is transmitted successfully or not.
        :param MATIs: maximum allowable transmission intervals.
        :return: solving the differential equation to find the new location, velocity, and acceleration.
        """
        y_ini_, time_, acc_value_ = self.value_update(time, y_ini, success_rate, MATIs, acc_vals) # time in ms
        Y_new, T_new = self.Diff_equatoin(acc_value_, y_ini_, time, time_)

        return T_new, Y_new

    def value_update(self, time, y_ini, s_rate, MATIs, acc_vals):
        '''
        Note: All the parameters are numpy arrays ---> easier implementation for pytorch ---> numpy to torch
        :param time: is the starting point of the differential equation --> in milliseconds
        :param y_ini: initial values computed from the last computation
        :param s_rate: is related to the packet transmission probability rate. when this is 1, means that the packet has
                       been transmitted successfully in the previous slot.
        :param MATIs: Is related to the Maximum Allowable Transmission Intervals (MATIs)
        :return: starting and ending point of the integral for ODE calculations, acceleration value, and the new initial
                 values. Upon successful reception of a control signal, the error 'e_u' should be reset to zero.
        -------------------------------------------------------------------------
        A quick explanation on the algorithm:
        With this algorithm, we are actually trying to achieve the following goals:
        1- finding the starting point and ending point needed for calculating the integral of ODE.
        2- finding the changes occurring in the acceleration profile. when we have a change in the acceleration, the
           integral bounds changes. We should calculate the integral first up to the starting point of the acceleration,
           taking into account that the error is zero, and then calculating the integral for the rest of the time up to
           the allowed MATI.
        3- We first check the success rate vector. If there is zero in one of the elements, the integration bounds will
           be [t, t+1]; meaning that we do the resource allocation again and again until all the vehicles transmit their
           control signals.
        4- when all the pairs finish transmitting their control signals (all the elements in the success rate become 1),
           then we can extend the integration bounds beyond one 1ms, but care must be taken since this can be very
           tricky. The integration bounds in this case would be from the current time until the min(delta_i, acc_time).
           meaning that normally the next time is the minimum of the MATIs, but it is also possible that any change in
           the acceleration profile happen between the MATIs. In these cases the integration bounds should also take
           them into account.
        '''

        # checking whether we need to do the scheduling for the upcoming slot or not?
        if np.sum(s_rate) != self.n_V2V_links:
            time_ = time + 1
        else:
            time_ = np.minimum(np.min(MATIs), self.next_acceleration_time(time))

        '''
        checking whether the control signal error must be set to zero or not? if the packet is not transmitted 
        successfully, then the 'e_u' will remain unchanged. upon successful transmission, the error must reset to zero.
        * Exception: In some cases that all the links have transmitted successfully, we can extend the integral 
        boundaries beyond one 1ms. in this cases, if we come across a change in the acceleration profile, up to the
        beginning point of the acceleration, the error is zero but after that point, the error will remain, since 
        technically, the information that has been sent from the vehicles, do not comply with the new information.
        '''
        for i in range(self.n_V2V_links):
            if (s_rate[i] == 1) and (acc_vals[i] == self.previous_acceleration_compute(time_)): # s_rate for the first link is always 1 ---> dynamic vehicle
                y_ini[self.Platoon.N_v*4 + i] = 0

        # For the last link, we always set the error to zero (last vehicle does not transmit to any vehicle)
        y_ini[self.Platoon.N_v * 4 + self.n_V2V_links] = 0
        # return y_ini, time_, self.current_acceleration_compute(time), Cntr
        return y_ini, time_, self.current_acceleration_compute(time)

    def Diff_equatoin(self, input_value, initial_value, start, end):
        """
        :param model: this is the model of the platoon
        :param input_value: acceleration profile of the platoon leader
        :return: the differential equation value
        """
        self.start = int(start)/1000
        self.end = int(end)/1000
        t_span = np.linspace(self.start, self.end, num=self.ODEScaling).reshape(-1)
        ode_out = solve_ivp(ode_platoon, [self.start, self.end], y0=initial_value, t_eval=t_span, method='DOP853',
                            args=(self.AA, self.BB, input_value))
        y_new = (ode_out.y).T
        t_new = (ode_out.t).T
        return y_new, t_new * 1000

    def current_acceleration_compute(self, val):
        '''
        :param val: time at which we wish to find the acceleration value for.
        :return: the acceleration value at the desired time
        ***
        this function is for computing the acceleration value of the desired time instant. hence this
        function is rather backward-looking. Another similar function is also implemented which is used for finding the
        next integration point which is forward-looking.
        '''
        where_ind = val - self.pert_intervals
        where_ind_first = np.where(where_ind >= 0)
        acc_val = where_ind_first[0][-1]

        return self.pert_values[acc_val]

    def next_acceleration_time(self, val):
        """
        :param val: what is the next upcoming acceleration time after 'val'?
        :return: the next acceleration value time coming after param 'val' :).
        """
        where_ind = val - self.pert_intervals
        where_ind_first = np.where(where_ind < 0)
        acc_ind = where_ind_first[0]

        return self.pert_intervals[acc_ind[0]] if len(acc_ind) != 0 else self.sim_time

    def previous_acceleration_compute(self, val):
        """
        :param val: what is the next upcoming acceleration time after 'val'?
        :return: the next acceleration value time coming after param 'val' :).
        """
        where_ind = val - self.pert_intervals
        where_ind_first = np.where(where_ind > 0)
        acc_ind = where_ind_first[0][-1]

        return self.pert_values[acc_ind]

    def String_Stability_slotted(self, total_y, total_t, Cntr, idx):
        '''
        :param total_y: Differential equation total outputs
        :param k: Big K matrix --> u = KX
        :return: String stability form the beginning of time slot 0; string stability of the last time slot
        '''
        SS_points = total_y.shape[0] - (self.ODEScaling-1) * int(Cntr[idx])
        self.String_Y = np.dot(self.Platoon.K, total_y[SS_points:,:4*self.size_platoon].transpose())
        self.String_time_ = total_t[SS_points:]

        return self.String_Y, self.String_time_

    def String_Stability_total(self, total_y, total_t):

        self.String_Y = np.dot(self.Platoon.K, total_y[:, :4 * self.size_platoon].transpose())

        return self.String_Y, total_t