import torch

class PolicyNN(torch.nn.Module):
    def __init__(self, n_input, n_output, n_hidden=64, continuous_actions=True,
                 value_network=True, shared_value_network=False):
        super(PolicyNN, self).__init__()
        self.value_network = value_network
        self.shared_value_network = shared_value_network
        self.feature_extractor = torch.nn.Sequential(torch.nn.Linear(n_input, n_hidden),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(n_hidden, n_hidden),
                                                     torch.nn.ReLU()
                                                    )
        self.policy_head = torch.nn.Linear(n_hidden, n_output)
        
        if self.value_network:
            # value head for baseline
            self.value_head = torch.nn.Linear(n_hidden, 1)
            if not self.shared_value_network:
                self.value_feature_extractor = torch.nn.Sequential(torch.nn.Linear(n_input, n_hidden),
                                                            torch.nn.ReLU(),
                                                            torch.nn.Linear(n_hidden, n_hidden),
                                                            torch.nn.ReLU()
                                                            )
        if continuous_actions:
            # state independent weight for log standard deviations initialized to 0.
            self.log_std = torch.nn.Parameter(torch.zeros(n_output))
            # self.log_std = torch.nn.Parameter(torch.full((n_output,), -0.5))  # std ≈ 0.6

    def forward(self, x):
        features = self.feature_extractor(x)
        policy_output = self.policy_head(features)
        if self.value_network:
            if not self.shared_value_network:
                value_features = self.value_feature_extractor(x)
                value_output = self.value_head(value_features)
            else:
                value_output = self.value_head(features)
            return policy_output, value_output
        else:
            return policy_output
