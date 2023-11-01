import numpy as np


class Platoon:
    def __init__(self, vehPairParams, N_v, Headway, v):

        self.platoon_params = [Headway, vehPairParams["na"], vehPairParams["kga"],
                               vehPairParams["kp"], vehPairParams["kd"]]
        self.N_v = N_v
        self.Headway = Headway
        self.v = v  # depends on the outage probability of the link
        self.A = []
        self.Bs = []
        self.Bc = []
        self.Br = []
        self.K = []


    def Create_platoon_Mats(self):

        Full_params = np.kron(np.array(self.platoon_params).reshape(1, len(self.platoon_params)),
                              np.ones(self.N_v).reshape(self.N_v, 1))

        # Vehicle 0 dynamics: Reference vehicle

        h = Full_params[0][0]
        etai = Full_params[0][1]
        kgi = Full_params[0][2]
        kp = Full_params[0][3]
        kd = Full_params[0][4]

        A00 = np.array([[0, 0, 0, 0],[0, 0, 1, 0],[0, 0, -1/etai, 0],[0, 0, 0, 0]])
        Bs0 = np.array([[0], [0], [kgi/etai], [0]])

        etai1 = etai
        kgi1 = kgi

        # Vehicle 1 dynamics: Platoon Leader (although the variable names are similar for the leader and the reference
        # vehicles, but care must be taken since for the heterogeneous cases, this can vary.

        h = Full_params[1][0]
        etai = Full_params[1][1]
        kgi = Full_params[1][2]
        kp = Full_params[1][3]
        kd = Full_params[1][4]
        gamma = (1-etai/etai1)*(1/(h*kgi))
        alpha = (kgi1/kgi)*(etai/etai1)*(1/h)

        A10 = np.array([[0, 1, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, gamma, 0]])
        A11 = np.array([[0, -1, -h, 0],[0, 0, 1, 0],[0, 0, -1/etai, 0],[0, 0, 0, -1/h]])
        Bs1 = np.array([[0], [0], [kgi/etai], [0]])
        Bc1 = np.array([[0], [0], [0], [alpha]])
        K10 = np.array([0, kd, 0, 0])
        K11 = np.array([kp, -kd, -kd*h, self.v])

        # Creating the big matrices for calculation
        A = np.block([[A00, np.zeros_like(A10)], [A10, A11]])
        Bs = np.block([[np.zeros_like(Bs1)], [Bs1]])
        Bc = np.block([[np.zeros_like(Bs1)], [np.zeros_like(Bs1)]])
        Br = np.block([[Bs0], [Bc1]])
        K = np.block([K10, K11])
        etai1 = etai
        kgi1 = kgi

        # Defining the dynamics of ith vehicle in the platoon
        for i in range(2, self.N_v):

            # parameter definition for the ith vehicles:
            h = Full_params[i][0]
            etai = Full_params[i][1]
            kgi = Full_params[i][2]
            kp = Full_params[i][3]
            kd = Full_params[i][4]
            gamma = (1 - etai / etai1) * (1 / (h * kgi))
            alpha = (kgi1 / kgi) * (etai / etai1) * (1 / h)

            # Creating the big matrices for the ith vehicles
            Aii1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, gamma, 0]])
            Aii = np.array([[0, -1, -h, 0], [0, 0, 1, 0], [0, 0, -1 / etai, 0], [0, 0, 0, -1 / h]])
            Bsi = np.array([[0], [0], [kgi / etai], [0]])
            Bci = np.array([[0], [0], [0], [alpha]])
            Kii1 = np.array([0, kd, 0, 0])
            Kii = np.array([kp, -kd, -kd * h, self.v])

            A = np.block([[A, np.zeros_like(A[:, 0:4])], [np.zeros_like(A[0:4, 0:(4*(i-1))]), Aii1, Aii]])
            Bs = np.block([[Bs, np.zeros_like(Bs[:, 0]).reshape(-1,1)], [np.zeros_like(Bs[0:4, :]), Bsi]])
            Bc = np.block([[Bc, np.zeros_like(Bc[:, 0]).reshape(-1,1)], [np.zeros([4, i-2]), Bci, np.zeros([4,1])]])
            if i-2 == 0:
                K = K.reshape([1, K.shape[0]])
            # Sorry for applying such tedious way of coding. 'K' changes the size in every loop, therefore K(:, 0:4)
            # tends to spit out the error that the array size does not match the entries. I needed to change the size
            # of the K to ,ake sure this error does not appear again. Think about a faster algorithm if you can.
            K = np.block([[K, np.zeros_like(K[:,0:4])], [np.zeros([1, (4*(i-1))]), Kii1, Kii]])
            Br = np.block([[Br],[np.zeros_like(Bsi)]])
            etai1 = etai

        self.A = A
        self.Bs = Bs
        self.Bc = Bc
        self.Br = Br
        self.K = K