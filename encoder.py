import torch.nn as nn
from torch import cat
from torch.linalg import norm
from RIRMetricsExperiments import get_pytorch_rir

class RoomToShoeboxEncoder(nn.Module):
    '''
    It's a simple encoder module.
    It takes in a batch of input features (mic_position (3d), src_position (3d), vertices(nx (8) + ny (8))) of shape = [batch_size, 22].
    It has an intermediate shoebox representation ((9,1) tensor = room_dim, mic_pos, src_pos)
    And finally outputs shoebox_rir and shoebox_rir_origin.

    Here, 8 2d vertices are used, but conceivably any amount of vertices is ok. Please initialize the input length to the right size.
    '''
    def __init__(self, input_size=22, hidden_sizes=[30,20], sample_rate=48000, max_order=10):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_sizes[1], 9),
            nn.Sigmoid()
        )
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order

    def forward(self, input):
        '''
        outputs shoebox_rir_batch, shoebox_origin_batch
        shapes torch.Size([4, 24000]) torch.Size([4])
        '''
        batch_size=input.size[0]
        
        z = self.NN(input)
        room_dimensions = z[:, 0:3]*20 # ARBITRARY NUMBER, HOW TO DO THIS DIFFERENTLY? USED A SIGMOID INSTEAD OF EXP...
        mic_position = z[:, 3:6]*room_dimensions
        source_position = z[:, 6:9]*room_dimensions

        # Forward pass through backpropagatable pytorch shoebox RIR calculation
        # Is there a way to do this in parallel?
        for i in range(batch_size):
            # print("Getting pytorch rir is slow, this is a bottleneck")
            shoebox_rir=get_pytorch_rir(room_dimensions[i],mic_position[i],source_position[i],self.sample_rate, max_order=self.max_order)
            if i==0: shoebox_rir_batch=shoebox_rir[None,:]
            else: shoebox_rir_batch=cat((shoebox_rir_batch,shoebox_rir[None,:]),0)

        # Get torch origins
        torch_distances = norm(mic_position-source_position, dim=1)
        shoebox_origin_batch = 40 + (self.sample_rate*torch_distances/self.sound_speed)

        return shoebox_rir_batch, shoebox_origin_batch


class GraphToShoeboxEncoder(nn.Module):
    def __init__(self, input_size=22, hidden_sizes=[30,20], sample_rate=48000, max_order=10):
        super().__init__()
        # self.NN = nn.Sequential(
        #     nn.Linear(input_size, hidden_sizes[0]),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden_sizes[1], 9),
        #     nn.Sigmoid()
        # )
        # RandomPool/DiffPool and GCN until fixed size
        # GCN and 2 layer MLP Alternating
        # MLP
        self.sample_rate=sample_rate
        self.sound_speed=343
        self.max_order=max_order

    def forward(self, input):
        '''
        outputs shoebox_rir_batch, shoebox_origin_batch
        shapes torch.Size([4, 24000]) torch.Size([4])
        '''
        # Use RandomPool/DiffPool and GCN until fixed size
        # Use GCN and 2 layer MLP Alternating
        # Use MLP

        # room_dimensions = z[:, 0:3]*20 # ARBITRARY NUMBER, HOW TO DO THIS DIFFERENTLY? USED A SIGMOID INSTEAD OF EXP...
        # mic_position = z[:, 3:6]*room_dimensions
        # source_position = z[:, 6:9]*room_dimensions

        # # Forward pass through backpropagatable pytorch shoebox RIR calculation
        # # Is there a way to do this in parallel?
        # for i in range(room_dimensions.size[0]):
        #     # print("Getting pytorch rir is slow, this is a bottleneck")
        #     shoebox_rir=get_pytorch_rir(room_dimensions[i],mic_position[i],source_position[i],self.sample_rate, max_order=self.max_order)
        #     if i==0: shoebox_rir_batch=shoebox_rir[None,:]
        #     else: shoebox_rir_batch=cat((shoebox_rir_batch,shoebox_rir[None,:]),0)

        # # Get torch origins
        # torch_distances = norm(mic_position-source_position, dim=1)
        # shoebox_origin_batch = 40 + (self.sample_rate*torch_distances/self.sound_speed)

        # return shoebox_rir_batch, shoebox_origin_batch