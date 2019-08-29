import torch.nn as nn
from utils import *


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.input_size = opt.input_size
        self.hidden_size = opt.encoder_hidden_size
        self.T = opt.T
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1)
        self.attn_layer_input = nn.Linear(self.T, 1)
        self.attn_layer_hidden = nn.Linear(2*self.hidden_size, self.T)
        self.attn_layer = nn.Linear(self.T, 1)

    def init_hidden(self, x):
        return x.data.new(1, x.size(0), self.hidden_size).zero_()

    def forward(self, input_data):
        input_weighted = input_data.data.new(input_data.size(0), self.T, self.input_size).zero_()
        input_encoded = input_data.data.new(input_data.size(0), self.T, self.hidden_size).zero_()

        hidden = self.init_hidden(input_data)
        cell = self.init_hidden(input_data)

        for t in range(self.T):
            hidden_part = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                                     cell.repeat(self.input_size, 1, 1).permute(1, 0, 2)), dim=2)
            x = torch.tanh(self.attn_layer_hidden(hidden_part.view(-1, self.hidden_size*2)) +
                           self.attn_layer_input(input_data.permute(0, 2, 1).contiguous().view(-1, self.T)))
            x = self.attn_layer(x)
            attn_weights = torch.softmax(x.view(-1, self.input_size), dim=-1)
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.T = opt.T
        self.encoder_hidden_size = opt.encoder_hidden_size
        self.decoder_hidden_size = opt.decoder_hidden_size
        self.attn_layer = nn.Sequential(nn.Linear(2*self.decoder_hidden_size+self.encoder_hidden_size, self.encoder_hidden_size),
                                        nn.Tanh(), nn.Linear(self.encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + 1, 1)
        self.fc.weight.data.normal_()
        self.fc_final = nn.Linear(self.encoder_hidden_size+self.decoder_hidden_size, 1)

    def init_hidden(self, x):
        return x.data.new(1, x.size(0), self.decoder_hidden_size).zero_()

    def forward(self, input_encoded, y_history):
        hidden = self.init_hidden(input_encoded)
        cell = self.init_hidden(input_encoded)

        for t in range(self.T-1):
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), input_encoded), dim=-1)
            x = torch.softmax(self.attn_layer(x.view(-1, 2*self.decoder_hidden_size+self.encoder_hidden_size)).view(-1, self.T), dim=-1)
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            if t < self.T-1:
                y_tild = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tild.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]
                cell = lstm_output[1]

        decoder_output = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return decoder_output


class LSNet(nn.Module):
    def __init__(self, opt):
        super(LSNet, self).__init__()
        self.m = opt.input_size
        self.P = opt.T
        self.hidR = opt.conv_size
        self.hidC = opt.conv_size
        self.kernel = opt.kernel_size
        self.out_size = opt.decoder_hidden_size
        self.conv1 = nn.Conv1d(in_channels=81, out_channels=self.hidC, kernel_size=2)
        self.lstm = nn.LSTM(self.hidC, self.hidR)
        self.linear = nn.Linear(self.hidR, opt.decoder_hidden_size)

    def forward(self, x):
        x = self.conv1(x).permute(2, 0, 1)
        _, (h, c) = self.lstm(x)
        return self.linear(h).view(-1, self.out_size)


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.lsnet = LSNet(opt)
        self.linear1 = nn.Linear(opt.encoder_hidden_size+2*opt.decoder_hidden_size, opt.decoder_hidden_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear2 = nn.Linear(opt.decoder_hidden_size, 1)

    def forward(self, x, y_history):
        input_weighted, input_encoded = self.encoder(x)
        decoder_output = self.decoder(input_encoded, y_history)
        # lsnet_output = self.lsnet(x.permute(0, 2, 1))
        # out = self.linear1(torch.cat((decoder_output, lsnet_output), dim=-1))
        return decoder_output


if __name__ == '__main__':
    opt = Configuration()
    a = torch.ones(16, 10, 81)
    b = torch.ones(16, 9)
    net = Model(opt)
    print(net(a, b).shape)
