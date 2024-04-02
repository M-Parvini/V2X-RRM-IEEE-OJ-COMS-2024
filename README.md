# CACC-RRM-Design-ICC-2023
Simulation code of the paper:
    "Joint Resource Allocation and String-Stable CACC Design with Multi-Agent Reinforcement Learning"

### If you want to cite: 
>M. Parvini, A. Gonzalez, A. Villamil, P. Schulz and G. Fettweis, “Joint Resource Allocation and String-Stable CACC Design with Multi-Agent Reinforcement Learning,” in Proceedings of 2023 International Conference on Communications (ICC 2023), Rome, Italy, May 2023.
---------------------------------------------------------------------------------------
### prerequisites:

    1) python 3.7 or higher
    2) PyTorch 1.7 or higher + CUDA
    3) It is recommended that the latest drivers be installed for the GPU.

***

### Algorithms that you can evaluate:

1. Federated Multi-Agent Reinforcement Learning
    + Set ***Algorithm = Hybrid, federated_communication = True, Train = True, Test = True***
    + You can also change the activation function to have either a linear or nonlinear function approximation model.
        * Set ***activation=Relu***, ***elu*** or ***leaky_relu*** for nonlinear function approximation or set ***activation=linear*** otherwise.
2. Decentralized Multi-Agent Reinforcement Learning
    + Set ***Algorithm = Hybrid, federated_communication = False, Train = True, Test = True***
    + You can also change the activation function to have either a linear or nonlinear function approximation model.
        * Set ***activation=Relu***, ***elu*** or ***leaky_relu*** for nonlinear function approximation or set ***activation=linear*** otherwise.
3. Sum-capacity optimization
    + Set ***Algorithm = Opt_non_fair, federated_communication = False, Train=False, Test = True***
4. Max-Min optimization
    + Set ***Algorithm = Opt_fair, federated_communication = False, Train = False, Test = True***
6. Random
    + Set ***Algorithm = random, federated_communication = False, Train = False, Test = True***


## Good Luck with your simulations!!!
