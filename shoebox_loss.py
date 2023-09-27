import torch
from torch.nn import MSELoss
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Shoebox_Loss(torch.nn.Module):
    def __init__(self, lambdas={"room_dim":1,"mic":1,"src":1,"mic_src_vector":1,"src_mic_vector":1,"absorption":1}, return_separate_losses=False):
        super().__init__()
        self.mse=torch.nn.MSELoss()
        self.lambdas=lambdas
        self.return_separate_losses=return_separate_losses
        print("Shoebox_Loss Initialized.")
    
    def forward(self, proposed_z_batch, label_z_batch, plot_i=0, return_separate_losses=False):
        '''
        args:
        proposed_z_batch: torch.tensor. shape: (batch_size, 10)
        label_z_batch: torch.tensor. shape: (batch_size, 10)
        plot_i: int, used for plotting in my training script
        '''

        assert(type(proposed_z_batch)==torch.tensor)
        assert(type(label_z_batch)==torch.tensor)
        assert(proposed_z_batch.shape==label_z_batch.shape)
        assert(proposed_z_batch.shape[1]==10)
        # batch_size=proposed_z_batch.shape[0]

        proposed_room_dimensions=proposed_z_batch[:,:3]
        proposed_mic_pos=proposed_z_batch[:,3:6]
        proposed_source_pos=proposed_z_batch[:,6:9]
        proposed_absorption=proposed_z_batch[:,9]

        target_room_dimensions=label_z_batch[:,:3]
        target_mic_pos=label_z_batch[:,3:6]
        target_source_pos=label_z_batch[:,6:9]
        target_mic_source_vector=target_source_pos-target_mic_pos
        target_source_mic_vector=target_mic_pos-target_source_pos
        target_absorption=label_z_batch[:,9]

        # Get room dimensions loss
        room_dimensions_loss=self.mse(proposed_room_dimensions, target_room_dimensions)
        # Get absorptions loss
        absorption_loss=self.mse(proposed_absorption, target_absorption)

        # Get mic and src losses which can have symmetries
        symmetries  =  [[ 1, 1, 1],
                        [ 1, 1,-1],
                        [ 1,-1, 1],
                        [-1, 1, 1],
                        [ 1,-1,-1],
                        [-1, 1,-1],
                        [-1,-1, 1],
                        [-1,-1,-1]]
        symmetries_uhhh=[[0.0,0.0,0.0],
                        [0.0,0.0,1.0],
                        [0.0,1.0,0.0],
                        [1.0,0.0,0.0],
                        [0.0,1.0,1.0],
                        [1.0,0.0,1.0],
                        [1.0,1.0,0.0],
                        [1.0,1.0,1.0]]
        symmetries=torch.tensor(symmetries, device=proposed_z_batch.device)
        symmetries_uhhh=torch.tensor(symmetries_uhhh, device=proposed_z_batch.device)        

        mic_loss=torch.tensor([0.0],device=proposed_z_batch.device)
        source_loss=torch.tensor([0.0],device=proposed_z_batch.device)
        mic_source_vector_loss=torch.tensor([0.0],device=proposed_z_batch.device)
        source_mic_vector_loss=torch.tensor([0.0],device=proposed_z_batch.device)

        for i in range(len(symmetries)):
            target_mic_pos_inverted=target_mic_pos*symmetries[i] + symmetries_uhhh[i]
            target_source_pos_inverted=target_source_pos*symmetries[i] + symmetries_uhhh[i]
            mic_source_vector_inverted=target_mic_source_vector*symmetries[i]
            source_mic_vector_inverted=target_source_mic_vector*symmetries[i]
            # target_mic_pos_inverted=torch.tensor([target_mic_pos[k] if symmetry[k] else 1.0-target_mic_pos[k] for k in range(3)])
            # target_source_pos_inverted=torch.tensor([target_source_pos[k] if symmetry[k] else 1.0-target_source_pos[k] for k in range(3)])
            # mic_source_vector_inverted=torch.tensor([target_mic_source_vector[k] if symmetry[k] else -target_mic_source_vector[k] for k in range(3)])
            # source_mic_vector_inverted=torch.tensor([target_source_mic_vector[k] if symmetry[k] else -target_source_mic_vector[k] for k in range(3)])

            mic_loss=mic_loss + (proposed_mic_pos-target_mic_pos_inverted).pow(2).sum().pow(0.2)
            source_loss=source_loss + (proposed_source_pos-target_source_pos_inverted).pow(2).sum().pow(0.2)
            mic_source_vector_loss=mic_source_vector_loss + (proposed_source_pos-(target_mic_pos+mic_source_vector_inverted)).pow(2).sum().pow(0.2)
            source_mic_vector_loss=source_mic_vector_loss + (proposed_mic_pos-(target_source_pos+source_mic_vector_inverted)).pow(2).sum().pow(0.2)

        if self.return_separate_losses or return_separate_losses :
            return room_dimensions_loss, mic_loss, source_loss, mic_source_vector_loss, source_mic_vector_loss, absorption_loss
        else:
            total_loss= room_dimensions_loss * self.lambdas["room_dim"]+\
                        mic_loss * self.lambdas["mic"]+\
                        source_loss * self.lambdas["src"]+\
                        mic_source_vector_loss * self.lambdas["mic_src_vector"]+\
                        source_mic_vector_loss * self.lambdas["src_mic_vector"]+\
                        absorption_loss * self.lambdas["absorption"]
            return total_loss

def main():
    room_dimension=torch.tensor([5.03,4.02,3.01])
    target_mic_pos=torch.rand(3)*room_dimension
    target_source_pos=torch.rand(3)*room_dimension

    mse=MSELoss()

    iterations=25000
    mic_loss_list=[]
    source_loss_list=[]
    mic_source_vector_loss_list=[]
    source_mic_vector_loss_list=[]
    proposed_mic_pos_list=[]
    proposed_source_pos_list=[]

    fig, axs = plt.subplots(1,4, figsize=(24,5))
    for i in range(iterations):
        proposed_mic_pos=torch.rand(3)*room_dimension
        proposed_source_pos=torch.rand(3)*room_dimension
        proposed_mic_pos[2]=target_mic_pos[2]
        proposed_source_pos[2]=target_source_pos[2]

        proposed_mic_pos_list.append(proposed_mic_pos)
        proposed_source_pos_list.append(proposed_source_pos)

        target_mic_source_vector=target_source_pos-target_mic_pos
        target_source_mic_vector=target_mic_pos-target_source_pos

        symmetries=[[True ,True ,True ],
           # [True ,True ,False],
           [True ,False,True ],
           [False,True ,True ],
           # [True ,False,False],
           # [False,True ,False],
           [False,False,True ],
           # [False,False,False]
           ]

        mic_loss=torch.tensor([0.0])
        source_loss=torch.tensor([0.0])
        mic_source_vector_loss=torch.tensor([0.0])
        source_mic_vector_loss=torch.tensor([0.0])

        for symmetry in symmetries:
            target_mic_pos_inverted=torch.tensor([target_mic_pos[k] if symmetry[k] else room_dimension[k]-target_mic_pos[k] for k in range(3)])
            target_source_pos_inverted=torch.tensor([target_source_pos[k] if symmetry[k] else room_dimension[k]-target_source_pos[k] for k in range(3)])
            mic_source_vector_inverted=torch.tensor([target_mic_source_vector[k] if symmetry[k] else -target_mic_source_vector[k] for k in range(3)])
            source_mic_vector_inverted=torch.tensor([target_source_mic_vector[k] if symmetry[k] else -target_source_mic_vector[k] for k in range(3)])

            mic_loss=mic_loss + torch.abs(proposed_mic_pos-target_mic_pos_inverted).pow(2).sum().pow(0.2)
            source_loss=source_loss + torch.abs(proposed_source_pos-target_source_pos_inverted).pow(2).sum().pow(0.2)
            mic_source_vector_loss=mic_source_vector_loss + torch.abs(proposed_source_pos-(target_mic_pos+mic_source_vector_inverted)).pow(2).sum().pow(0.2)
            source_mic_vector_loss=source_mic_vector_loss + torch.abs(proposed_mic_pos-(target_source_pos+source_mic_vector_inverted)).pow(2).sum().pow(0.2)

        mic_loss_list.append(mic_loss)
        source_loss_list.append(source_loss)
        mic_source_vector_loss_list.append(mic_source_vector_loss)
        source_mic_vector_loss_list.append(source_mic_vector_loss)

    mic_loss_list=torch.stack(mic_loss_list)
    source_loss_list=torch.stack(source_loss_list)
    mic_source_vector_loss_list=torch.stack(mic_source_vector_loss_list)
    source_mic_vector_loss_list=torch.stack(source_mic_vector_loss_list)
    proposed_mic_pos_list=torch.stack(proposed_mic_pos_list)
    proposed_source_pos_list=torch.stack(proposed_source_pos_list)

    axs[0].set_title('Mic position symmetries loss')
    axs[1].set_title('Source position symmetries loss')
    axs[2].set_title('Mic-Source Vector symmetries loss')
    axs[3].set_title('Source-Mic Vector symmetries loss')

    cmap = plt.get_cmap('viridis')

    scatter0=axs[0].scatter(proposed_mic_pos_list[:,0].numpy(),proposed_mic_pos_list[:,1].numpy(), c=mic_loss_list.flatten().numpy(), cmap=cmap)
    scatter1=axs[1].scatter(proposed_source_pos_list[:,0].numpy(),proposed_source_pos_list[:,1].numpy(), c=source_loss_list.flatten().numpy(), cmap=cmap)
    scatter2=axs[2].scatter(proposed_source_pos_list[:,0].numpy(),proposed_source_pos_list[:,1].numpy(), c=mic_source_vector_loss_list.flatten().numpy(), cmap=cmap)
    scatter3=axs[3].scatter(proposed_mic_pos_list[:,0].numpy(),proposed_mic_pos_list[:,1].numpy(), c=source_mic_vector_loss_list.flatten().numpy(), cmap=cmap)

    cbar0=plt.colorbar(scatter0,ax=axs[0])
    cbar1=plt.colorbar(scatter1,ax=axs[1])
    cbar2=plt.colorbar(scatter2,ax=axs[2])
    cbar3=plt.colorbar(scatter3,ax=axs[3])

    cbar0.set_label('mic to target mic distance')
    cbar1.set_label('source to target source distance')
    cbar2.set_label('source to target source distance')
    cbar3.set_label('mic to target mic distance')

    target_mic_pos=target_mic_pos.numpy()
    target_source_pos=target_source_pos.numpy()

    x=[target_mic_pos[0],target_mic_pos[0],room_dimension[0].item()-target_mic_pos[0],room_dimension[0].item()-target_mic_pos[0]]
    y=[target_mic_pos[1],room_dimension[1].item()-target_mic_pos[1],target_mic_pos[1],room_dimension[1].item()-target_mic_pos[1]]
    axs[0].scatter(x,y, c='red', marker='x', label='Target mic')
    axs[0].scatter(target_source_pos[0],target_source_pos[1], c='red', marker='o', label='Target source')

    axs[1].scatter(target_mic_pos[0],target_mic_pos[1], c='red', marker='x', label='Target mic')
    x=[target_source_pos[0],target_source_pos[0],room_dimension[0].item()-target_source_pos[0],room_dimension[0].item()-target_source_pos[0]]
    y=[target_source_pos[1],room_dimension[1].item()-target_source_pos[1],target_source_pos[1],room_dimension[1].item()-target_source_pos[1]]
    axs[1].scatter(x,y, c='red', marker='o', label='Target source')


    axs[2].scatter(target_mic_pos[0],target_mic_pos[1], c='red', marker='x', label='Target mic')
    src_x=[target_mic_pos[0]+target_mic_source_vector[0],
    target_mic_pos[0]+target_mic_source_vector[0],
    target_mic_pos[0]-target_mic_source_vector[0],
    target_mic_pos[0]-target_mic_source_vector[0]]
    src_y=[target_mic_pos[1]+target_mic_source_vector[1],
    target_mic_pos[1]-target_mic_source_vector[1],
    target_mic_pos[1]+target_mic_source_vector[1],
    target_mic_pos[1]-target_mic_source_vector[1]]
    axs[2].scatter(src_x,src_y, c='red', marker='o', label='Target source')

    axs[3].scatter(target_source_pos[0],target_source_pos[1], c='red', marker='o', label='Target source')
    mic_x=[target_source_pos[0]+target_source_mic_vector[0],
    target_source_pos[0]+target_source_mic_vector[0],
    target_source_pos[0]-target_source_mic_vector[0],
    target_source_pos[0]-target_source_mic_vector[0]]
    mic_y=[target_source_pos[1]+target_source_mic_vector[1],
    target_source_pos[1]-target_source_mic_vector[1],
    target_source_pos[1]+target_source_mic_vector[1],
    target_source_pos[1]-target_source_mic_vector[1]]
    axs[3].scatter(mic_x,mic_y, c='red', marker='x', label='Target mic')


    axs[0].add_patch(Rectangle((0,0),room_dimension[0].item(),room_dimension[1].item(), edgecolor='black', facecolor='none', lw=5, alpha=1))
    axs[1].add_patch(Rectangle((0,0),room_dimension[0].item(),room_dimension[1].item(), edgecolor='black', facecolor='none', lw=5, alpha=1))
    axs[2].add_patch(Rectangle((0,0),room_dimension[0].item(),room_dimension[1].item(), edgecolor='black', facecolor='none', lw=5, alpha=1))
    axs[3].add_patch(Rectangle((0,0),room_dimension[0].item(),room_dimension[1].item(), edgecolor='black', facecolor='none', lw=5, alpha=1))

    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel('y (m)')
    axs[2].set_xlabel('x (m)')
    axs[2].set_ylabel('y (m)')
    axs[3].set_xlabel('x (m)')
    axs[3].set_ylabel('y (m)')

    axs[0].axhline(room_dimension[1].item()/2, color='red', ls=':')
    axs[1].axhline(room_dimension[1].item()/2, color='red', ls=':')
    axs[2].axhline(target_mic_pos[1], color='red', ls=':')
    axs[3].axhline(target_source_pos[1], color='red', ls=':')

    axs[0].axvline(room_dimension[0].item()/2, color='red', ls=':')
    axs[1].axvline(room_dimension[0].item()/2, color='red', ls=':')
    axs[2].axvline(target_mic_pos[0], color='red', ls=':')
    axs[3].axvline(target_source_pos[0], color='red', ls=':')

    axs[0].set_xlim([0,room_dimension[0].item()])
    axs[0].set_ylim([0,room_dimension[1].item()])
    axs[1].set_xlim([0,room_dimension[0].item()])
    axs[1].set_ylim([0,room_dimension[1].item()])
    axs[2].set_xlim([0,room_dimension[0].item()])
    axs[2].set_ylim([0,room_dimension[1].item()])
    axs[3].set_xlim([0,room_dimension[0].item()])
    axs[3].set_ylim([0,room_dimension[1].item()])

    axs[0].legend(loc='lower left')
    axs[1].legend(loc='lower left')
    axs[2].legend(loc='lower left')
    axs[3].legend(loc='lower left')

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()