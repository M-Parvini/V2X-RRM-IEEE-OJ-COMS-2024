from base_funcs.TD3_agent import Agent as TD3_Agent

class TD3_v2v_agent(TD3_Agent):
    def __init__(self, id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                 Fed_Comm, folder):
        super().__init__(id, gamma, mem_size, epsilon, taus, batch_size, actor_net, critic_net, full_data, algorithm,
                         Fed_Comm, folder)

    def train(self):

        self.learn()
        self.clone_model_paramenter(self.actor_model.parameters(), self.local_actor_model)
        self.clone_model_paramenter(self.critic1_model.parameters(), self.local_critic1_model)
        self.clone_model_paramenter(self.critic2_model.parameters(), self.local_critic2_model)
