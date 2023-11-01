import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TD3_CriticNetwork(nn.Module):
    def __init__(self, critic_lr, input_dims, C_fc_dims_list, activation, squash, initialize, n_actions):
        super(TD3_CriticNetwork, self).__init__()
        #Initialize
        self.state_size = input_dims + n_actions
        self.action_size = n_actions
        self.activation = activation
        self.squashing_function = squash
        self.init_type = initialize
        self.critic_lr = critic_lr
        self.output_layer_init_std = 0.0001

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        last_Layer_Size = inputSize
        if C_fc_dims_list is not None:
            nh = len(C_fc_dims_list)
            self.layers.append(nn.Linear(inputSize, C_fc_dims_list[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(C_fc_dims_list[i - 1], C_fc_dims_list[i]))
            last_Layer_Size = C_fc_dims_list[nh - 1]

        self.q = nn.Linear(last_Layer_Size, 1)
        # self.action_passthrough_layer = nn.Linear(self.state_size, self.action_size)

        # initialise layer weights
        if self.init_type == "kaiming":
            for i in range(0, len(self.layers)):
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
                bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
                self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
            nn.init.kaiming_normal_(self.q.weight, nonlinearity=self.activation)
            bias_ini = 1. / np.sqrt(self.q.weight.data.size()[0])
            self.q.bias.data.uniform_(-bias_ini, bias_ini)
        elif self.init_type == "phil_tabor":
            f3 = 0.003
            for i in range(0, len(self.layers)):
                bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
                self.layers[i].weight.data.uniform_(-bias_ini, bias_ini)
                self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
            self.q.weight.data.uniform_(-f3, f3)
            self.q.bias.data.uniform_(-f3, f3)
        else:
            raise ValueError("Unknown init_type " + str(self.init_type))

        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        # implement forward
        negative_slope = 0.1

        x = T.cat((state, action), dim=1)
        num_hidden_layers = len(self.layers)

        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))

        state_action_value = self.q(x)

        return state_action_value

class TD3_ActorNetwork(nn.Module):
    def __init__(self, actor_lr, input_dims, A_fc_dims_list, activation, squash, initialize, n_actions):
        super(TD3_ActorNetwork, self).__init__()
        # Initialize
        self.state_size = input_dims
        self.action_size = n_actions
        self.activation = activation
        self.squashing_function = squash
        self.init_type = initialize
        self.output_layer_init_std = 0.001
        self.actor_lr = actor_lr

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        last_Layer_Size = inputSize
        if A_fc_dims_list is not None:
            nh = len(A_fc_dims_list)
            self.layers.append(nn.Linear(inputSize, A_fc_dims_list[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(A_fc_dims_list[i - 1], A_fc_dims_list[i]))
            last_Layer_Size = A_fc_dims_list[nh - 1]
        self.action_output_layer = nn.Linear(last_Layer_Size, self.action_size)

        # initialise layer weights
        if self.init_type == "kaiming":
            for i in range(0, len(self.layers)):
                nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
                bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
                self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
            nn.init.kaiming_normal_(self.action_output_layer.weight, nonlinearity=self.activation)
            bias_ini = 1. / np.sqrt(self.action_output_layer.weight.data.size()[0])
            self.action_output_layer.bias.data.uniform_(-bias_ini, bias_ini)
        elif self.init_type == "phil_tabor":
            f3 = 0.003
            for i in range(0, len(self.layers)):
                bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
                self.layers[i].weight.data.uniform_(-bias_ini, bias_ini)
                self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
            self.action_output_layer.weight.data.uniform_(-f3, f3)
            self.action_output_layer.bias.data.uniform_(-f3, f3)
        else:
            raise ValueError("Unknown init_type " + str(self.init_type))

        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = state
        negative_slope = 0.1
        num_hidden_layers = len(self.layers)

        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            elif self.activation == "elu":
                x = F.elu(self.layers[i](x))
            else:
                raise ValueError("Unknown activation function " + str(self.activation))

        action_params = T.sigmoid(self.action_output_layer(x))
        # action_params += self.action_passthrough_layer(state)

        return action_params


# class TD3_CriticNetwork(nn.Module):
#     def __init__(self, critic_lr, input_dims, C_fc_dims_list, activation, squash, initialize, n_actions):
#         super(TD3_CriticNetwork, self).__init__()
#         #Initialize
#         self.state_size = input_dims
#         self.action_size = n_actions
#         self.activation = activation
#         self.squashing_function = squash
#         self.init_type = initialize
#         self.critic_lr = critic_lr
#         self.output_layer_init_std = 0.0001
#
#         # create layers
#         self.layers = nn.ModuleList()
#         inputSize = self.state_size
#         last_Layer_Size = inputSize
#         if C_fc_dims_list is not None:
#             nh = len(C_fc_dims_list)
#             self.layers.append(nn.Linear(inputSize, C_fc_dims_list[0]))
#             for i in range(1, nh):
#                 self.layers.append(nn.Linear(C_fc_dims_list[i - 1], C_fc_dims_list[i]))
#             last_Layer_Size = C_fc_dims_list[nh - 1]
#
#         self.action_value = nn.Linear(self.action_size, C_fc_dims_list[-1])
#         self.q = nn.Linear(last_Layer_Size, 1)
#         # self.action_passthrough_layer = nn.Linear(self.state_size, self.action_size)
#
#         # initialise layer weights
#         for i in range(0, len(self.layers)):
#             if self.init_type == "kaiming":
#                 nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=self.activation)
#                 bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
#                 self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
#             elif self.init_type == "normal":
#                 nn.init.normal_(self.layers[i].weight, std=1)
#                 bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
#                 self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
#                 # ToDo: if you were to use normal distribution, play with the std value to find the best combination
#             elif self.init_type == "phil_tabor":
#                 bias_ini = 1. / np.sqrt(self.layers[i].weight.data.size()[0])
#                 self.layers[i].weight.data.uniform_(-bias_ini, bias_ini)
#                 self.layers[i].bias.data.uniform_(-bias_ini, bias_ini)
#             else:
#                 raise ValueError("Unknown init_type " + str(self.init_type))
#
#         if self.init_type == "phil_tabor":
#             f3 = 0.003
#             self.q.weight.data.uniform_(-f3, f3)
#             self.q.bias.data.uniform_(-f3, f3)
#         else:
#             nn.init.normal_(self.q.weight, std=self.output_layer_init_std)
#             bias_ini = 1. / np.sqrt(self.q.weight.data.size()[0])
#             self.q.bias.data.uniform_(-bias_ini, bias_ini)
#
#         f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
#         self.action_value.weight.data.uniform_(-f4, f4)
#         self.action_value.bias.data.uniform_(-f4, f4)
#         # nn.init.zeros_(self.layers[-1].bias)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=critic_lr, weight_decay=0.01)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#
#         self.to(self.device)
#
#     def forward(self, state, action):
#         # implement forward
#         negative_slope = 0.1
#
#         x = state
#         num_hidden_layers = len(self.layers)
#
#         for i in range(0, num_hidden_layers):
#             if self.activation == "relu":
#                 x = F.relu(self.layers[i](x))
#                 if i == num_hidden_layers-1: # second layer
#                     action_value = F.relu(self.action_value(action))
#                     x = T.add(x, action_value)
#             elif self.activation == "leaky_relu":
#                 x = F.leaky_relu(self.layers[i](x), negative_slope)
#                 if i == num_hidden_layers-1: # second layer
#                     action_value = F.leaky_relu(self.action_value(action), negative_slope)
#                     x = T.add(x, action_value)
#             elif self.activation == "elu":
#                 x = F.elu(self.layers[i](x))
#                 if i == num_hidden_layers-1: # second layer
#                     action_value = F.elu(self.action_value(action), negative_slope)
#                     x = T.add(x, action_value)
#             else:
#                 raise ValueError("Unknown activation function " + str(self.activation))
#
#         state_action_value = self.q(x)
#
#         return state_action_value