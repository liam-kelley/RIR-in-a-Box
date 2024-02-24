import torch
import torch.nn as nn
import torch.nn.parallel
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

################### MESH2IR MESHNET ###################

class data_for_meshnet():
    def __init__(self, pos, edge_index, batch):
        self.pos = pos
        self.edge_index = edge_index
        self.batch = batch

class MESH_NET(nn.Module):
    '''
    baseline lifted from MESH2IR
    '''
    def __init__(self):
        super(MESH_NET,self).__init__()
        self.feature_dim = 3
        self.conv1 = GCNConv(self.feature_dim, 32)
        self.pool1 = TopKPooling(32, ratio=0.6)
        self.conv2 = GCNConv(32, 32) #(32, 64)
        self.pool2 = TopKPooling(32, ratio=0.6) #64, ratio=0.6)
        self.conv3 = GCNConv(32, 32) #(64, 128)
        self.pool3 = TopKPooling(32, ratio=0.6) #(128, ratio=0.6)
        # self.item_embedding = torch.nn.Embedding(num_embeddings=df.item_id.max() +1, embedding_dim=self.feature_dim)
        self.lin1 = torch.nn.Linear(64, 16) #(256, 128)
        self.lin2 = torch.nn.Linear(16, 8) #(128, 64)
        # self.lin3 = torch.nn.Linear(8, 1) #(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(16) #(128)
        self.bn2 = torch.nn.BatchNorm1d(8) #(64)
        self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU() 
        print("MESH_NET initialized")   
  
    def forward(self, data):
        x, edge_index, batch = data.pos, data.edge_index, data.batch
        # x = self.item_embedding(x)
        # x = x.squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ ,_= self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
     
        x, edge_index, _, batch, _,_ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _,_ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        # x = self.lin2(x)
        # x = self.act2(x)      
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin2(x)).squeeze(1)
        return x

################### MESH2IR GAN ###################

class COND_NET(nn.Module):
    '''
    baseline lifted from MESH2IR
    '''
    def __init__(self):
        super(COND_NET, self).__init__()
        self.t_dim = 14
        self.c_dim = 10
        self.fc = nn.Linear(self.t_dim, self.c_dim, bias=True)
        self.relu = nn.PReLU()

    def encode(self, full_embed):
        x = self.relu(self.fc(full_embed))
        return x

    def forward(self, full_embed):
        c_code = self.encode(full_embed)
        return c_code #, mu, logvar

def upBlock4(in_planes, out_planes):
    '''
    baseline lifted from MESH2IR
    '''
    kernel_length  = 41
    stride = 4
    block = nn.Sequential(
        nn.ConvTranspose1d(in_planes,out_planes,kernel_size=kernel_length,stride=stride, padding=19,output_padding=1),
        nn.BatchNorm1d(out_planes),
        nn.PReLU())
    return block

def upBlock2(in_planes, out_planes):
    '''
    baseline lifted from MESH2IR
    '''
    kernel_length  = 41
    stride = 2
    block = nn.Sequential(
        nn.ConvTranspose1d(in_planes,out_planes,kernel_size=kernel_length,stride=stride, padding=20,output_padding=1),
        nn.BatchNorm1d(out_planes),
        nn.PReLU())
    return block

class STAGE1_G(nn.Module):
    '''
    baseline lifted from MESH2IR
    forward inputs are text_embedding and mesh_embed.
    mesh embed is the batch output of MESH_NET.
    text_embedding is the batch [[source_x, source_y, source_z, mic_x, mic_y, mic_z], ...]
    '''
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = 256 * 8
        self.ef_dim = 10
        self.define_module()

    def define_module(self):
        kernel_length  = 41
        ninput = self.ef_dim
        ngf = self.gf_dim
        self.cond_net = COND_NET()
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 16, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.PReLU())

        # ngf x 16 -> ngf/2 x 64
        self.upsample1 = upBlock4(ngf, ngf // 2)
        # -> ngf/4 x 256
        self.upsample2 = upBlock4(ngf // 2, ngf // 4)
        # -> ngf/8 x 1024
        self.upsample3 = upBlock4(ngf // 4, ngf // 8)
        # -> ngf/16 x 4096
        self.upsample4 = upBlock2(ngf // 8, ngf // 16)
        self.upsample5 = upBlock2(ngf // 16, ngf // 16)
        # -> 1 x 4096
        self.RIR = nn.Sequential(
            nn.ConvTranspose1d(ngf // 16,1,kernel_size=kernel_length,stride=1, padding=20),
            nn.Tanh())

    def forward(self, text_embedding,mesh_embed):
        full_embed= torch.cat((mesh_embed, text_embedding), 1)
        c_code = self.cond_net(full_embed)

        h_code = self.fc(c_code)

        h_code = h_code.view(-1, self.gf_dim, 16)
        # print("h_code 1 ",h_code.size())
        h_code = self.upsample1(h_code)
        # print("h_code 2 ",h_code.size())
        h_code = self.upsample2(h_code)
        # print("h_code 3 ",h_code.size())
        h_code = self.upsample3(h_code)
        # print("h_code 4 ",h_code.size())
        h_code = self.upsample4(h_code)
        h_code = self.upsample5(h_code)
        # print("h_code 5 ",h_code.size())
        # # state size 3 x 64 x 64
        fake_RIR = self.RIR(h_code)
        # print("fake_RIR ",fake_RIR.size())
        # # return None, fake_RIR, mu, logvar
        # #print("generator ", text_embedding.size())
        # return None, fake_RIR, text_embedding #c_code
        return None, fake_RIR, c_code

################### MESH2IR FULL MODEL ###################
    
class MESH2IR_FULL(nn.Module):
    '''
    combines both parts of the MESH2IR model for simple evaluation or training.
    '''
    def __init__(self, mesh_net, net_G):
        super(MESH2IR_FULL, self).__init__()
        self.mesh_net = mesh_net
        self.net_G = net_G

    def forward(self, x, edge_index, batch, batch_oracle_mic_pos, batch_oracle_src_pos):
        data = data_for_meshnet(x, edge_index, batch) # the pretrained mesh_net we use uses a data struct for input data.
        mesh_embed = self.mesh_net(data)
        text_embedding = torch.cat((batch_oracle_src_pos, batch_oracle_mic_pos), dim=1)
        _, rir, _ = self.net_G(text_embedding, mesh_embed)
        rir = rir.squeeze(1)
        return rir
