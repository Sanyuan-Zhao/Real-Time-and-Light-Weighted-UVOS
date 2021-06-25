import torch
from torch import nn
from torch.autograd import Variable
from networks.convLSTM import ConvLSTMCell
import random


class RNN_Conv_Cell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, p_TD):
        super(RNN_Conv_Cell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # divide exactly
        self.bias = bias

        self.p_TD = p_TD

        self.spatial =nn.Conv2d(in_channels=self.input_dim,
                                  out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

        self.temporal = nn.Conv2d(in_channels=self.input_dim * 2,
                                  out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):
        rate = random.random()

        c = cur_state
        spatial_x = input_tensor
        temporal_x = input_tensor.detach() if rate < self.p_TD else input_tensor
        temporal_in = torch.cat((c, temporal_x), dim=1)
        spatial_out = torch.tanh(self.spatial(spatial_x))
        temporal_out = torch.sigmoid(self.temporal(temporal_in))
        spatial_temporal = spatial_out * temporal_out
        return spatial_temporal, temporal_out


    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())


class RNN_Conv(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, p_TD,
                 batch_first=False, bias=True, return_all_layers=False):
        super(RNN_Conv, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.p_TD = p_TD
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(RNN_Conv_Cell(input_size=(self.height, self.width),
                                      input_dim=cur_input_dim,
                                      hidden_dim=self.hidden_dim[i],
                                      kernel_size=self.kernel_size[i],
                                      bias=self.bias,
                                      p_TD=self.p_TD))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        #print('hidden_state:',hidden_state)
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor.permute(1, 0, 2, 3, 4)

        if hidden_state is not None:
            pass
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=c)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(c)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    print('start')
    net = LSTM_Conv(input_size=(30, 54),
		            input_dim=256,
                    hidden_dim=[256, 256],
		            kernel_size=(5, 5),
		            num_layers=2,
		            p_TD=0.3,
		            batch_first=True,
		            bias=True,
		            return_all_layers=True)
    #print(net)
    net.cuda()
    batch_image = Variable(torch.randn(1, 2, 256, 30, 54)).cuda()
    layer_output_list, last_state_list = net(batch_image)
    print('layer_output_list:',len(layer_output_list))
    print('last_state_list:',len(last_state_list))

    #y1,y2,y3,y4,y5 = net.forward(batch_image)