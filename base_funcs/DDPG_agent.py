from copy import deepcopy
import numpy as np
import os
import torch
import torch as T
from Networks.noise import OUActionNoise
from torch.autograd import Variable
import torch.nn.functional as F
from Networks.buffer import ReplayBuffer
from Utils import soft_update, hard_update

class Agent:
    def __init__(self, id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                 Fed_Comm, folder):

        self.actor_model = deepcopy(actor_net)
        self.target_actor_model = deepcopy(actor_net)
        self.critic1_model = deepcopy(critic_net)
        self.target_critic1_model = deepcopy(critic_net)
        self.critic2_model = deepcopy(critic_net)
        self.target_critic2_model = deepcopy(critic_net)
        self.id = id
        self.clip_grad = 10
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = 0.1
        self.eps_decay = 5e-5
        self.tau = taus
        self.batch_size = batch_size
        self.full_data = full_data * batch_size
        self.n_actions = self.actor_model.action_size
        self.n_states = self.actor_model.state_size
        self.min_power = 0
        self.algorithm = algorithm
        self.folder = folder
        self.Fed_Comm = Fed_Comm
        self.create_model_path()

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        self.memory = ReplayBuffer(mem_size, self.n_states, self.n_actions, algorithm)

        hard_update(self.actor_model, self.target_actor_model)
        self.target_actor_model.eval()
        hard_update(self.critic1_model, self.target_critic1_model)
        self.target_critic1_model.eval()
        hard_update(self.critic2_model, self.target_critic2_model)
        self.target_critic2_model.eval()

        # saving the gradients of the local models and the server
        self.local_actor_model = deepcopy(list(self.actor_model.parameters()))
        self.local_critic1_model = deepcopy(list(self.critic1_model.parameters()))
        self.local_critic2_model = deepcopy(list(self.critic2_model.parameters()))

        self.server_Trx_actor_grad = deepcopy(list(self.actor_model.parameters()))
        self.server_Trx_critic1_grad = deepcopy(list(self.critic1_model.parameters()))
        self.server_Trx_critic2_grad = deepcopy(list(self.critic2_model.parameters()))

        self.pre_local_actor_grad = deepcopy(list(self.actor_model.parameters()))
        self.pre_local_critic1_grad = deepcopy(list(self.critic1_model.parameters()))
        self.pre_local_critic2_grad = deepcopy(list(self.critic2_model.parameters()))

    def set_passthrough_resenet_weight_biases(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_model.action_passthrough_layer

        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.actor_model.device)

        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.actor_model.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update(self.actor_model, self.target_actor_model)

    def choose_action(self, observation, option):
        #  Choosing action: combination of DDPG and DQN
        with torch.no_grad():
            # epsilon Greedy; similar to DQN policy
            rnd = np.random.uniform(low=0, high=1)
            if rnd < self.epsilon:
                continuous_acts = torch.tensor(np.random.uniform(-1, 1, self.n_actions))
            else:
                # select maximum action
                observation_ = np.expand_dims(observation, axis=0)
                state = T.tensor(observation_, dtype=T.float).to(self.actor_model.device)
                continuous_acts = self.actor_model.forward(state).to(self.actor_model.device)

            # Adding noise to the selected actions
            continuous_acts = continuous_acts.cpu().data.numpy().flatten()
            if option == 'train':
                continuous_acts = continuous_acts + self.noise()

        return continuous_acts

    def learn(self):

        assert self.memory.mem_cntr >= self.full_data

        states, actions, rewards, states_, done = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor_model.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor_model.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor_model.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor_model.device)
        done = T.tensor(done).to(self.actor_model.device)

        target_actions = self.target_actor_model.forward(states_)
        critic_value_ = self.target_critic1_model.forward(states_, target_actions)
        critic_value = self.critic1_model.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic1_model.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic1_model.optimizer.step()

        self.actor_model.optimizer.zero_grad()
        actor_loss = -self.critic1_model.forward(states, self.actor_model.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor_model.optimizer.step()

        soft_update(self.critic1_model, self.target_critic1_model, self.tau[0])
        soft_update(self.critic2_model, self.target_critic2_model, self.tau[0])
        soft_update(self.actor_model, self.target_actor_model, self.tau[1])

        self.decrement_epsilon()

        return critic_loss, actor_loss

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_decay \
            if self.epsilon > self.eps_min else self.eps_min

    def set_parameters(self, server_actor_model, server_critic1_model, server_critic2_model):
        # actor
        for old_actor_param, server_actor_param, local_actor_param in zip(self.actor_model.parameters(),
                                                                          server_actor_model.parameters(),
                                                                          self.local_actor_model):
            old_actor_param.data = server_actor_param.data.clone()
            local_actor_param.data = server_actor_param.data.clone()
            if(server_actor_param.grad != None):
                if(old_actor_param.grad == None):
                    old_actor_param.grad = torch.zeros_like(server_actor_param.grad)

                if(local_actor_param.grad == None):
                    local_actor_param.grad = torch.zeros_like(server_actor_param.grad)

                old_actor_param.grad.data = server_actor_param.grad.data.clone()
                local_actor_param.grad.data = server_actor_param.grad.data.clone()

        # critic 1
        for old_critic1_param, server_critic1_param, local_critic1_param in zip(self.critic1_model.parameters(),
                                                                                server_critic1_model.parameters(),
                                                                                self.local_critic1_model):
            old_critic1_param.data = server_critic1_param.data.clone()
            local_critic1_param.data = server_critic1_param.data.clone()
            if(server_critic1_param.grad != None):
                if(old_critic1_param.grad == None):
                    old_critic1_param.grad = torch.zeros_like(server_critic1_param.grad)

                if(local_critic1_param.grad == None):
                    local_critic1_param.grad = torch.zeros_like(server_critic1_param.grad)

                old_critic1_param.grad.data = server_critic1_param.grad.data.clone()
                local_critic1_param.grad.data = server_critic1_param.grad.data.clone()

        # critic 2
        for old_critic2_param, server_critic2_param, local_critic2_param in zip(self.critic2_model.parameters(),
                                                                                server_critic2_model.parameters(),
                                                                                self.local_critic2_model):
            old_critic2_param.data = server_critic2_param.data.clone()
            local_critic2_param.data = server_critic2_param.data.clone()
            if(server_critic2_param.grad != None):
                if(old_critic2_param.grad == None):
                    old_critic2_param.grad = torch.zeros_like(server_critic2_param.grad)

                if(local_critic2_param.grad == None):
                    local_critic2_param.grad = torch.zeros_like(server_critic2_param.grad)

                old_critic2_param.grad.data = server_critic2_param.grad.data.clone()
                local_critic2_param.grad.data = server_critic2_param.grad.data.clone()

    def get_actor_parameters(self):
        return self.actor_model.parameters()

    def get_critic1_parameters(self):
        return self.critic1_model.parameters()

    def get_critic2_parameters(self):
        return self.critic2_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
            if (param.grad != None):
                if (clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()

        return clone_param

    def create_model_path(self):
        if self.Fed_Comm:
            self.intermediary_model_path = os.path.join(self.folder, "agent_models")
            if not os.path.exists(self.intermediary_model_path):
                os.makedirs(self.intermediary_model_path)
        else:
            self.intermediary_model_path = os.path.join(self.folder, "agent_models")
            if not os.path.exists(self.intermediary_model_path):
                os.makedirs(self.intermediary_model_path)

    def save_agent_model(self):
        torch.save(self.actor_model, os.path.join(self.intermediary_model_path, "actor_model_" + str(self.id) + ".pt"))
        torch.save(self.critic1_model, os.path.join(self.intermediary_model_path, "critic1_model_" + str(self.id) + ".pt"))
        torch.save(self.critic2_model, os.path.join(self.intermediary_model_path, "critic2_model_" + str(self.id) + ".pt"))
        torch.save(self.target_actor_model, os.path.join(self.intermediary_model_path, "target_actor_model_" + str(self.id) + ".pt"))
        torch.save(self.target_critic1_model, os.path.join(self.intermediary_model_path, "target_critic1_model_" + str(self.id) + ".pt"))
        torch.save(self.target_critic2_model, os.path.join(self.intermediary_model_path, "target_critic2_model_" + str(self.id) + ".pt"))

    def load_agent_model(self):
        model_actor_path = os.path.join(self.intermediary_model_path, "actor_model_" + str(self.id) + ".pt")
        model_target_actor_path = os.path.join(self.intermediary_model_path,
                                               "target_actor_model_" + str(self.id) + ".pt")
        model_critic1_path = os.path.join(self.intermediary_model_path, "critic1_model_" + str(self.id) + ".pt")
        model_critic2_path = os.path.join(self.intermediary_model_path, "critic2_model_" + str(self.id) + ".pt")

        model_target_critic1_path = os.path.join(self.intermediary_model_path, "target_critic1_model_" + str(self.id) + ".pt")
        model_target_critic2_path = os.path.join(self.intermediary_model_path, "target_critic2_model_" + str(self.id) + ".pt")

        self.actor_model = torch.load(model_actor_path)
        self.target_actor_model = torch.load(model_target_actor_path)
        self.critic1_model = torch.load(model_critic1_path)
        self.critic2_model = torch.load(model_critic2_path)
        self.target_critic1_model = torch.load(model_target_critic1_path)
        self.target_critic2_model = torch.load(model_target_critic2_path)
