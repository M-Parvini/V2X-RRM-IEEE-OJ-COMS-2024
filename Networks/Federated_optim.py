from torch.optim import Optimizer

class Federated_Optimizer(Optimizer):
    def __init__(self, params, lr = 0.01, hyper_lr = 0.01,  L = 0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr,hyper_lr= hyper_lr, L = L)
        super(Federated_Optimizer, self).__init__(params, defaults)

    def step(self, server_grads, pre_grads, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, server_grad, pre_grad in zip(group['params'],server_grads, pre_grads):
                if(server_grad.grad != None and pre_grad.grad != None):
                    p.data = p.data - group['lr'] * (p.grad.data + group['hyper_lr'] * server_grad.grad.data - pre_grad.grad.data)
                else:
                     p.data = p.data - group['lr'] * p.grad.data
        return loss