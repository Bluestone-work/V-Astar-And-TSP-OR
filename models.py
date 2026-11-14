import torch
import torch.nn as nn
import torch.nn.functional as F
class DuelingDQN_with_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, lstm_layers=1):
        super(DuelingDQN_with_LSTM, self).__init__()
        self.distance_angle_fc = nn.Linear(2, 64)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 7x7x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 4x4x32

        self.lstm_input_dim = 32 * 4 * 4 + 64

        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)

        lstm_output_dim = hidden_dim * 2

        self.value_fc = nn.Linear(lstm_output_dim, 128)
        self.value_output = nn.Linear(128, 1)

        self.advantage_fc = nn.Linear(lstm_output_dim, 128)
        self.advantage_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(1)

        batch_size, time_steps, state_dim = x.size()

        grid_states = x[:, :, 2:].reshape(batch_size * time_steps, 1, 7, 7)

        grid_features = F.relu(self.conv1(grid_states))
        grid_features = F.relu(self.conv2(grid_features))

        grid_features = grid_features.view(batch_size * time_steps, -1)

        distance_angle = x[:, :, :2].reshape(batch_size * time_steps, 2)
        distance_angle = F.relu(self.distance_angle_fc(distance_angle))

        combined_features = torch.cat([grid_features, distance_angle], dim=1)

        combined_features = combined_features.view(batch_size, time_steps, -1)
        lstm_out, (h_n, c_n) = self.lstm(combined_features)

        lstm_out_last = lstm_out[:, -1, :]

        value = F.relu(self.value_fc(lstm_out_last))
        value = self.value_output(value)

        advantage = F.relu(self.advantage_fc(lstm_out_last))
        advantage = self.advantage_output(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DuelingDQN_with_uniLSTM(nn.Module):
    """Dueling Q network with a unidirectional temporal encoder."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.distance_angle_fc = nn.Linear(2, 64)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTM(32 * 4 * 4 + 64, hidden_dim, batch_first=True)
        self.value_fc = nn.Linear(hidden_dim, 128); self.value_output = nn.Linear(128, 1)
        self.advantage_fc = nn.Linear(hidden_dim, 128); self.advantage_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        b, t, _ = x.shape
        g = x[:, :, 2:].reshape(b * t, 1, 7, 7)
        g = F.relu(self.conv2(F.relu(self.conv1(g)))).reshape(b * t, -1)
        d = F.relu(self.distance_angle_fc(x[:, :, :2].reshape(b * t, 2)))
        z, _ = self.lstm(torch.cat([g, d], 1).reshape(b, t, -1))
        z = z[:, -1]
        value = self.value_output(F.relu(self.value_fc(z)))
        advantage = self.advantage_output(F.relu(self.advantage_fc(z)))
        return value + advantage - advantage.mean(1, keepdim=True)


class PlainDQN_with_LSTM(nn.Module):
    """Unidirectional LSTM encoder with a standard (non-dueling) head."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.distance_angle_fc = nn.Linear(2, 64)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTM(32 * 4 * 4 + 64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 128); self.q_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        b, t, _ = x.shape
        g = x[:, :, 2:].reshape(b * t, 1, 7, 7)
        g = F.relu(self.conv2(F.relu(self.conv1(g)))).reshape(b * t, -1)
        d = F.relu(self.distance_angle_fc(x[:, :, :2].reshape(b * t, 2)))
        z, _ = self.lstm(torch.cat([g, d], 1).reshape(b, t, -1))
        return self.q_output(F.relu(self.fc(z[:, -1])))


class PlainDQN_with_BiLSTM(DuelingDQN_with_LSTM):
    """Identical BiLSTM/CNN encoder with a single action-value head."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__(input_dim, output_dim, hidden_dim)
        del self.value_fc, self.value_output, self.advantage_fc, self.advantage_output
        self.fc = nn.Linear(hidden_dim * 2, 128)
        self.q_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, t, _ = x.shape
        g = x[:, :, 2:].reshape(b * t, 1, 7, 7)
        g = F.relu(self.conv2(F.relu(self.conv1(g)))).reshape(b * t, -1)
        d = F.relu(self.distance_angle_fc(x[:, :, :2].reshape(b * t, 2)))
        encoded, _ = self.lstm(torch.cat([g, d], 1).reshape(b, t, -1))
        return self.q_output(F.relu(self.fc(encoded[:, -1])))


class PlainDQN(nn.Module):
    """Feed-forward convolutional DQN used for the architecture ablation."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.distance_angle_fc = nn.Linear(2, 64)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 4 * 4 + 64, 128); self.q_output = nn.Linear(128, output_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        b, t, _ = x.shape
        x = x[:, -1]
        g = F.relu(self.conv2(F.relu(self.conv1(x[:, 2:].reshape(b, 1, 7, 7))))).reshape(b, -1)
        d = F.relu(self.distance_angle_fc(x[:, :2]))
        return self.q_output(F.relu(self.fc(torch.cat([g, d], 1))))
