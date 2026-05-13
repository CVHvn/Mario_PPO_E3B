class Memory():
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rewards_int = []
        self.rewards_int_norm = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def save(self, state, action, reward, reward_int, reward_int_norm, next_state, done, logit, value, value_int):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.rewards_int.append(reward_int)
        self.rewards_int_norm.append(reward_int_norm)
        self.dones.append(done)
        self.logits.append(logit)
        self.values.append(value)
        self.values_int.append(value_int)

    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rewards_int = []
        self.rewards_int_norm = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def get_data(self):
        return self.states, self.actions, self.next_states, self.rewards, self.rewards_int, self.rewards_int_norm, self.dones, self.logits, self.values, self.values_int