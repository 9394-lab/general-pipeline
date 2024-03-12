import torch
from torch import nn







class MGRU(nn.Module):
    def __init__(self, input_channels_list, hidden_channels,
                    output_channels_list=[1, 1, 3]):
        nn.Module.__init__(self)
        self.name = 'MGRU'
        self.save_name = self.name
        self.input_channels_list = input_channels_list
        self.output_channels_list = output_channels_list


        self.input_gru_list = nn.ModuleList(
            [nn.GRU(input_channels, hidden_channels, num_layers=1, batch_first=True) for input_channels in input_channels_list]
        )

        self.tower_list =   nn.ModuleList([
            nn.Sequential(nn.GRU(hidden_channels*len(input_channels_list), hidden_channels, num_layers=1, batch_first=True),
                          nn.ReLU(),
                          nn.Linear(hidden_channels, output_channels))
            for output_channels in output_channels_list
        ])


    def forward(self, x):


        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]
        fea_num = x.shape[3]


        x = x.view(batch_size*num_nodes, time_len, fea_num)

        cumsum = 0
        input_gru_out = []
        for g, in_channels in zip(self.input_gru_list, self.input_channels_list):
            tmp_in = x[...,cumsum:(cumsum+in_channels)]
            cumsum += in_channels
            input_gru_out.append(g(tmp_in))

        input_gru_out = torch.cat(input_gru_out, dim=-1)

        tower_out_list = []
        for tower in self.tower_list:
            tower_out_list.append(tower(input_gru_out))

        return torch.cat(tower_out_list, dim=-1).view(batch_size, num_nodes, time_len, -1)