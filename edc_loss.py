import torch
import matplotlib.pyplot as plt
from typing import Union

class EDC_Loss(torch.nn.Module):
    def __init__(self, deemphasize_early_reflections=True,plot=False, plot_every=2, edr=False):
        super().__init__()
        self.mse=torch.nn.MSELoss(reduction='sum')
        self.loss_division=20000 #(often, there are around 20000 samples. Since actual length is variable, we can't do reduction='mean', we'll just divide by a fixed constant like 20000)
        self.plot=plot
        self.plot_every=plot_every
        self.deemphasize_early_reflections=deemphasize_early_reflections
        self.edr=edr
        print("EDC_Loss Initialized. Using loss mode : ", end='')
        if self.edr: print("EDR", end='')
        else: print("EDC", end='')
        if self.deemphasize_early_reflections: print(" with deemphasized early reflections")
        else: print("")
    
    def forward(self, shoebox_rir_batch : Union[list,torch.Tensor], shoebox_origin_batch : torch.Tensor, label_rir_batch : Union[list,torch.Tensor], label_origin_batch : torch.Tensor, plot_i=0):
        '''
        args:
            shoebox_rir_batch: list of torch.Tensor, each tensor is a shoebox rir
            shoebox_origin_batch: torch.Tensor, each element is the origin of the corresponding shoebox rir
            label_rir_batch: list of torch.Tensor, each tensor is a label rir
            label_origin_batch: torch.Tensor, each element is the origin of the corresponding label rir
            device: str, device to use
            plot_i: int, used for plotting in my training script
        '''
        if isinstance(shoebox_rir_batch, list): assert(type(shoebox_rir_batch[0])==torch.Tensor)
        if isinstance(label_rir_batch, list): assert(type(label_rir_batch[0])==torch.Tensor)
        assert(len(shoebox_rir_batch)==len(label_rir_batch) == shoebox_origin_batch.shape[0] == label_origin_batch.shape[0])
        batch_size=shoebox_origin_batch.shape[0]

        # crop rirs to begin from origins
        new_shoebox_rir_batch=[]
        new_label_rir_batch=[]
        for i in range(batch_size):
            origin_shoebox=int(shoebox_origin_batch[i].item())
            origin_label=int(label_origin_batch[i].item())
            new_shoebox_rir_batch.append(shoebox_rir_batch[i][max(0,origin_shoebox-10):])
            new_label_rir_batch.append(label_rir_batch[i][max(0,origin_label-10):])
            if  self.plot and not self.edr and i==0 and plot_i%self.plot_every ==0:
                plt.figure()
                plt.title('EDC Loss')
                plt.plot(abs(new_shoebox_rir_batch[0].cpu().detach().numpy()) / max(abs(new_shoebox_rir_batch[0].cpu().detach().numpy())), c='blue', alpha=0.3)
                plt.plot(abs(new_label_rir_batch[0].cpu().detach().numpy()) / max(abs(new_label_rir_batch[0].cpu().detach().numpy())), c='darkorange', alpha=0.3)

        shoebox_rir_batch=torch.nn.utils.rnn.pad_sequence(new_shoebox_rir_batch, batch_first=True).to(shoebox_origin_batch.device)
        label_rir_batch=torch.nn.utils.rnn.pad_sequence(new_label_rir_batch, batch_first=True).to(shoebox_origin_batch.device)
        
        # Free memory
        for i in range(batch_size):
            new_shoebox_rir_batch[i] = None
            new_label_rir_batch[i] = None
        del new_shoebox_rir_batch
        del new_label_rir_batch

        if self.edr:
            # compute stft magnitude on 7 bands (nfft//2 + 1)
            shoebox_rir_batch=torch.stft(shoebox_rir_batch, n_fft=13, return_complex=True)
            label_rir_batch=torch.stft(label_rir_batch, n_fft=13, return_complex=True)
            shoebox_rir_batch = (shoebox_rir_batch.real**2) + (shoebox_rir_batch.imag**2)
            label_rir_batch = (label_rir_batch.real**2) + (label_rir_batch.imag**2)
        else:
            # compute powers
            shoebox_rir_batch=torch.pow(shoebox_rir_batch,2)
            label_rir_batch=torch.pow(label_rir_batch,2)
        
        # do cumulative sum
        shoebox_rir_batch=torch.cumsum(torch.flip(shoebox_rir_batch, dims=[-1]), dim=-1) # Cumulative sum starting from the end
        label_rir_batch=torch.cumsum(torch.flip(label_rir_batch,dims=[-1]), dim=-1) # Cumulative sum starting from the end
        
        # normalize
        if self.edr : sb_normalizer=shoebox_rir_batch[...,-1,None] ; label_normalizer=label_rir_batch[...,-1,None]
        else: sb_normalizer=shoebox_rir_batch[:,-1].unsqueeze(1) ; label_normalizer=label_rir_batch[:,-1].unsqueeze(1)
        shoebox_rir_batch=shoebox_rir_batch/sb_normalizer
        label_rir_batch=label_rir_batch/label_normalizer
        del sb_normalizer, label_normalizer

        # pad to same length
        if shoebox_rir_batch.shape[-1] < label_rir_batch.shape[-1]:
            shoebox_rir_batch = torch.nn.functional.pad(shoebox_rir_batch, (label_rir_batch.shape[-1]-shoebox_rir_batch.shape[-1], 0)) # padding from the beginning because we flipped
        elif shoebox_rir_batch.shape[-1] > label_rir_batch.shape[-1]:
            label_rir_batch = torch.nn.functional.pad(label_rir_batch, (shoebox_rir_batch.shape[-1]-label_rir_batch.shape[-1], 0)) # padding from the beginning because we flipped

        if self.plot and not self.edr and plot_i%self.plot_every ==0:
            if self.deemphasize_early_reflections: ls=':'
            else: ls='-'
            plt.plot(torch.flip(shoebox_rir_batch[0].cpu().detach(),dims=[0]).numpy(), c='darkblue', alpha=1 , ls=ls)
            plt.plot(torch.flip(label_rir_batch[0].cpu().detach(),dims=[0]).numpy(), c='orange', alpha=1 , ls=ls)

        if self.deemphasize_early_reflections :
            get_rid_of_early_reflections=torch.arange(2000,0, -1).to(shoebox_rir_batch.device)
            get_rid_of_early_reflections=get_rid_of_early_reflections/len(get_rid_of_early_reflections)
            shoebox_rir_batch[..., -len(get_rid_of_early_reflections):] =   shoebox_rir_batch[..., -len(get_rid_of_early_reflections):]*get_rid_of_early_reflections[..., :shoebox_rir_batch.shape[-1]]
            label_rir_batch[..., -len(get_rid_of_early_reflections):] =   label_rir_batch[..., -len(get_rid_of_early_reflections):]*get_rid_of_early_reflections[..., :shoebox_rir_batch.shape[-1]]
            del get_rid_of_early_reflections

            if self.plot and not self.edr and plot_i%self.plot_every ==0:
                plt.plot(torch.flip(shoebox_rir_batch[0].cpu().detach(),dims=[0]).numpy(), c='darkblue', alpha=1)
                plt.plot(torch.flip(label_rir_batch[0].cpu().detach(),dims=[0]).numpy(), c='orange', alpha=1)

        if self.plot and self.edr and plot_i%self.plot_every ==0 :
            fig, ax = plt.subplots(figsize=(17,9.8))
            fig.suptitle("Multi-Resolution STFT Visualisation")
            ax.set_title('EDR Difference')
            quadmesh = ax.pcolormesh((torch.abs(torch.flip(shoebox_rir_batch[0],dims=(-1,))-torch.flip(label_rir_batch[0],dims=(-1,)))).cpu().numpy(), cmap='inferno')
            # Add the colorbar for this specific subplot
            cbar=fig.colorbar(quadmesh, ax=ax)
            cbar.set_label("Energy Difference")
            ax.set_xlabel("time in sample bins")
            ax.set_ylabel("frequency bands")
            plt.show()

        loss=self.mse(shoebox_rir_batch, label_rir_batch) / self.loss_division
        del shoebox_rir_batch, label_rir_batch
        
        return loss