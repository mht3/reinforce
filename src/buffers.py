class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.values = []

    def add(self, state, action, reward, value=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if value is not None:
            self.values.append(value)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.returns.clear()
        self.values.clear()