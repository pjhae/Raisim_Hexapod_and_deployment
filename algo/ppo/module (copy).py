import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from torch.nn.utils import weight_norm


class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape


class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, input_size, output_size):
        super(MLP, self).__init__()
        
        self.activation = nn.ReLU()

        modules = [nn.Linear(input_size, shape[0]), self.activation]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation)
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
         
    def forward(self, x):
        return self.architecture(x)


class CNN(nn.Module):
    def __init__(self, shape, input_size, output_size):
        super(CNN, self).__init__()
        
        self.conv_hist = weight_norm(nn.Conv1d(1, 4, kernel_size=3, stride=1, dilation=18))
        self.conv_leg_joints = weight_norm(nn.Conv1d(4, 16, kernel_size=3, stride=3, dilation=1))
        self.conv_legs = weight_norm(nn.Conv1d(16, 64, kernel_size=6, stride=1, dilation=1))
        
        self.fc1 = weight_norm(nn.Linear(3, 16))
        self.fc2 = weight_norm(nn.Linear(16+64, output_size))
        
        self.activation = nn.ReLU()
        
        self.down_conv = nn.Sequential(
        # 1, 54
            self.conv_hist, self.activation,
        # 4, 18
            self.conv_leg_joints, self.activation,
        # 16, 6
            self.conv_legs, self.activation
        # 64, 1
        )
        
        self.init_weights()
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def init_weights(self):
        self.conv_hist.weight.data.normal_(0, 1e-2)
        self.conv_leg_joints.weight.data.normal_(0, 1e-2)
        self.conv_legs.weight.data.normal_(0, 1e-2)
        self.fc1.weight.data.normal_(0, 1e-2)
        self.fc2.weight.data.normal_(0, 1e-2)
        
    def forward(self, x):
        y = self.down_conv(torch.unsqueeze(x[:, :-3], dim=1))
        y = torch.flatten(y, start_dim=1)
        z = self.activation(self.fc1(x[:, -3:]))
        return self.fc2(torch.cat((y, z), dim=1))


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.distribution = None
        if type(init_std) == float:
            self.std = nn.Parameter(init_std * torch.ones(dim))
        elif type(init_std) == list:
            self.std = nn.Parameter(torch.tensor(init_std))
        else:
            raise ValueError(f"Unsupported init_std type: {type(init_std)}")

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std
