import torch.nn as nn
import torch
from utils import Configuration


class FirstStage(nn.Module):
    def __init__(self, opt):
        super(FirstStage, self).__init__()
        self.input_size = opt.input_size
        self.hidden_size = opt.FS_hidden_size
        self.T = opt.T
        self.lstm_layer = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size)
        self.linear_hidden = nn.Linear(self.hidden_size*2, self.T)
        self.linear_input = nn.Linear(self.T-1, self.T)
        self.attn_layer = nn.Linear(self.T, 1)

    def init_hidden(self, x):  # batch_size * T-1 * input_size -> 1 * batch_size * hidden_size
        return x.data.new(1, x.size(0), self.hidden_size).zero_()

    def forward(self, input_data):  # batch_size * T-1 * input_size
        hidden = self.init_hidden(input_data)  # 1 * batch_size * hidden_size
        cell = self.init_hidden(input_data)  # 1 * batch_size * hidden_size

        input_weighted = input_data.data.new(input_data.size(0), self.T-1, self.input_size).zero_()
        input_encoded = input_data.data.new(input_data.size(0), self.T-1, self.hidden_size).zero_()

        for t in range(self.T-1):
            hidden_cell_part = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                                          cell.repeat(self.input_size, 1, 1).permute(1, 0, 2)), dim=-1)  # batch_size * input_size * (2*hidden_size)

            attn = self.attn_layer(torch.tanh(self.linear_hidden(hidden_cell_part.view(-1, self.hidden_size*2)) +
                                                     self.linear_input(input_data.permute(0, 2, 1).contiguous().view(-1, self.T-1))))
            # (batch_size * input_size) * 1
            attn_weights = torch.softmax(attn.view(-1, self.input_size), dim=-1)
            # batch_size * input_size
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            # batch_size * input size
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            input_encoded[:, t, :] = hidden
            input_weighted[:, t, :] = weighted_input
        # batch_size * T-1 * input_size batch_size * T-1 * hidden_size
        return input_weighted, input_encoded


class SecondStage(nn.Module):
    def __init__(self, opt):
        super(SecondStage, self).__init__()
        self.input_size = opt.input_size
        self.hidden_size = opt.SS_hidden_size
        self.T = opt.T
        self.linear_hidden = nn.Linear(opt.FS_hidden_size*2, self.T)
        self.linear_FS = nn.Linear(self.T, self.T)
        self.attn_layer = nn.Linear(self.T, 1)
        self.lstm_layer = nn.LSTM(input_size=opt.input_size, hidden_size=opt.SS_hidden_size, num_layers=1)

    def init_hidden(self, x):
        return x.data.new(1, x.size(0), self.hidden_size).zero_()

    def forward(self, FS_output, y_history):
        input_weighted = FS_output.data.new(FS_output.size(0), self.T-1, self.input_size).zero_()
        input_encoded = FS_output.data.new(FS_output.size(0), self.T-1, self.hidden_size).zero_()

        hidden = self.init_hidden(FS_output)
        cell = self.init_hidden(FS_output)

        for t in range(self.T-1):
            hidden_cell_part = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                                          cell.repeat(self.input_size, 1, 1).permute(1, 0, 2)), dim=-1)
            #  batch_size * input_size * (2*hidden_size)

            x_hat_part = torch.cat((FS_output.permute(0, 2, 1), y_history[:, t].view(-1, 1, 1).repeat(1, self.input_size, 1)), dim=-1)
            attn = self.attn_layer(torch.tanh(self.linear_hidden(hidden_cell_part)+self.linear_FS(x_hat_part)))
            attn_weights = torch.softmax(attn.view(-1, self.input_size), dim=-1)
            weighted_input = torch.mul(attn_weights, FS_output[:, t, :])
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
        self.encoder_hidden_size = opt.SS_hidden_size
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
            x = torch.cat((hidden.repeat(self.T-1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T-1, 1, 1).permute(1, 0, 2), input_encoded), dim=-1)
            x = torch.softmax(self.attn_layer(x.view(-1, 2*self.decoder_hidden_size+self.encoder_hidden_size)).view(-1, self.T-1), dim=-1)
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]
            if t < self.T-1:
                y_tild = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tild.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]
                cell = lstm_output[1]

        decoder_output = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return decoder_output


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.encoder_FS = FirstStage(opt)
        self.encoder_SS = SecondStage(opt)
        self.decoder = Decoder(opt)
        self.linear1 = nn.Linear(opt.SS_hidden_size+2*opt.decoder_hidden_size, opt.decoder_hidden_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear2 = nn.Linear(opt.decoder_hidden_size, 1)

    def forward(self, x, y_history):
        input_weighted, input_encoded = self.encoder_FS(x)
        input_weighted, input_encoded = self.encoder_SS(input_weighted, y_history)
        decoder_output = self.decoder(input_encoded, y_history)
        # lsnet_output = self.lsnet(x.permute(0, 2, 1))
        # out = self.linear1(torch.cat((decoder_output, lsnet_output), dim=-1))
        return decoder_output

if __name__ == '__main__':
    opt = Configuration()
    test = Model(opt)
    inp = torch.ones(16, 9, 81)
    y_history = torch.zeros(16, 9)
    print(test(inp, y_history).shape)