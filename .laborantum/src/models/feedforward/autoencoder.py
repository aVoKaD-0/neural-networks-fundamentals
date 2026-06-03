import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        ...
        super().__init__()
        ## YOUR CODE HERE
        # -- placeholder start --
        encoder_layers = []
        for index in range(len(channels) - 2):
            encoder_layers.append(torch.nn.Linear(channels[index], channels[index + 1]))
            encoder_layers.append(activation())
        encoder_layers.append(torch.nn.Linear(channels[-2], channels[-1]))

        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_channels = channels[::-1]
        for index in range(len(decoder_channels) - 2):
            decoder_layers.append(torch.nn.Linear(decoder_channels[index], decoder_channels[index + 1]))
            decoder_layers.append(activation())
        decoder_layers.append(torch.nn.Linear(decoder_channels[-2], decoder_channels[-1]))

        self.decoder = torch.nn.Sequential(*decoder_layers)
        # -- placeholder end --
        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = signal
        ## YOUR CODE HERE
        # -- placeholder start --
        res = res.reshape([res.shape[0], -1])
        res = self.encoder(res)
        res = self.decoder(res)
        # -- placeholder end --
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        ## YOUR CODE HERE
        # -- placeholder start --
        reconstruction = self.__forward_kernel(batch['data']['image'])
        batch['signals'] = {'reconstruction': reconstruction}
        # -- placeholder end --
        if 'signals' not in batch:
            batch['signals'] = {'reconstruction': batch['data']['image']}
        return batch
