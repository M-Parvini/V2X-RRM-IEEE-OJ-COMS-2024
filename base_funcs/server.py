import torch
import os
from copy import deepcopy

class Server:
    def __init__(self, actor_model, critic_model, num_users, Fed_comm, folder):
        # Set up the main attributes
        self.Fed_Comm = Fed_comm
        self.folder = folder
        self.server_actor_model = deepcopy(actor_model)
        self.server_critic_model = deepcopy(critic_model)
        self.users = []
        self.num_users = num_users
        self.create_server_path() if (actor_model != None) else None

    def send_parameters(self):
        # Server sends its parameters to the UEs
        for user in self.users:
            user.set_parameters(self.server_actor_model, self.server_critic_model)

    def aggregate_parameters(self):
        # actor
        for param in self.server_actor_model.parameters():
            param.data = torch.zeros_like(param.data)
            if (param.grad != None):
                param.grad.data = torch.zeros_like(param.grad.data)

        # critic
        for param in self.server_critic_model.parameters():
            param.data = torch.zeros_like(param.data)
            if (param.grad != None):
                param.grad.data = torch.zeros_like(param.grad.data)

        for user in self.users:
            self.add_parameters(user, 1 / self.num_users)
            # self.add_grad(user, user.train_samples / total_train)

    def add_parameters(self, user, ratio):
        # Creating the server's model from the agents models
        # Actor nets
        for server_param, user_param in zip(self.server_actor_model.parameters(), user.get_actor_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
            if (user_param.grad != None):
                if (server_param.grad == None):
                    server_param.grad = torch.zeros_like(user_param.grad)
                server_param.grad.data = server_param.grad.data + user_param.grad.data.clone() * ratio

        # critic nets
        for server_param, user_param in zip(self.server_critic_model.parameters(), user.get_critic1_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
            if (user_param.grad != None):
                if (server_param.grad == None):
                    server_param.grad = torch.zeros_like(user_param.grad)
                server_param.grad.data = server_param.grad.data + user_param.grad.data.clone() * ratio

    def create_server_path(self):
        if self.Fed_Comm:
            self.server_path = os.path.join(self.folder, "server_models")
            if not os.path.exists(self.server_path):
                os.makedirs(self.server_path)
        else:
            self.server_path = os.path.join(self.folder, "server_models")
            if not os.path.exists(self.server_path):
                os.makedirs(self.server_path)

    def save_server_model(self):
        torch.save(self.server_actor_model, os.path.join(self.server_path, "server_actor_model" + ".pt"))
        torch.save(self.server_critic_model, os.path.join(self.server_path, "server_critic_model" + ".pt"))

    def load_server_model(self):
        model_actor_path = os.path.join(self.server_path, "server_actor_model" + ".pt")
        model_critic_path = os.path.join(self.server_path, "server_critic_model" + ".pt")

        # using the cpu for inference
        self.server_actor_model = torch.load(model_actor_path)
        self.server_critic_model = torch.load(model_critic_path)
