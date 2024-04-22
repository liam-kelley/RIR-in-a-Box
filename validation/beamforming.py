import torch
import numpy as np
from datasets.GWA_3DFRONT.dataset import GWA_3DFRONT_Dataset
from datasets.ValidationDataset.dataset import HL2_Dataset
from torch.utils.data import DataLoader
from models.utility import load_all_models_for_inference, inference_on_all_models
from losses.rir_losses import EnergyDecay_Loss, MRSTFT_Loss, AcousticianMetrics_Loss, DRR_Loss
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import glob
import soundfile as sf
from tools.gcc_phat import gcc_phat
from librosa import load
from scipy.signal import fftconvolve
import torch.nn.functional as F

def metric_accuracy_mesh2ir_vs_rirbox_HL2(model_config : str, validation_csv : str, validation_iterations=0,
                                                 SCALE_MESH2IR_BY_ITS_ESTIMATED_STD = True, # If True, cancels out the std normalization used during mesh2ir's training
                                                 SCALE_MESH2IR_GWA_SCALING_COMPENSATION = True, # If true, cancels out the scaling compensation mesh2ir learned from the GWA dataset during training.
                                                 MESH2IR_USES_LABEL_ORIGIN = False,
                                                 RESPATIALIZE_RIRBOX = False, # This both activates the respaitialization of the rirbox and the start from ir onset
                                                 ISM_MAX_ORDER = 18
                                                 ):
    ''' Validation of the metric accuracy of the MESH2IR and RIRBOX models on the HL2 dataset.'''

    print("Starting metric accuracy validation for model: ", model_config.split("/")[-1].split(".")[0],end="\n\n")

    mesh2ir, rirbox, config, DEVICE = load_all_models_for_inference(model_config,
                                                                    START_FROM_IR_ONSET=RESPATIALIZE_RIRBOX,
                                                                    ISM_MAX_ORDER=ISM_MAX_ORDER)

    # data
    dataset=HL2_Dataset(csv_file=validation_csv, load_hl2_array=True, )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=1, pin_memory=False,
                            collate_fn=HL2_Dataset.custom_collate_fn)
    print("")

    # Get wav file paths
    wavs = glob.glob("datasets/Small_Timit/*.wav")
    # choose a seed
    np.random.seed(42)
    # shuffle signals
    np.random.shuffle(wavs)

    fs = 16000
    c = 343

    my_list=[]
    with torch.no_grad():
        # iterate over the dataset
        iterations=0
        for x_batch, edge_index_batch, batch_indexes, label_rirs, label_origins, mic_pos_batch, src_pos_batch in tqdm(dataloader, desc="Sound source spatialization validation"): 
            mic_pos_1 = mic_pos_batch.squeeze() - torch.tensor([0.0-d/2, 0, 0.0])
            mic_pos_2 = mic_pos_batch.squeeze() - torch.tensor([0.0+d/2, 0, 0.0])
            mic_pos_1 = mic_pos_1.unsqueeze(0)
            mic_pos_2 = mic_pos_2.unsqueeze(0)
            arr_pos = mic_pos_batch.squeeze().cpu().numpy()
            src_loc_from_spherical = lambda dist, azimuth : (dist * np.r_[np.cos(azimuth), np.sin(azimuth), 0]) + arr_pos

            # import speech signal
            signal, fs = load(path=wavs[iterations%len(wavs)], sr=fs, mono=True, duration=3.7)

            
            rirs_mesh2ir=[]
            rirs_rirbox=[]
            origins_mesh2ir=[]
            origins_rirbox=[]
            distances=[]
            # for both mic pos. (TODO implement multichannel backpropagatable ISM)
            for mic_pos in [mic_pos_1, mic_pos_2]:
                # Get RIRS for our models
                rir_mesh2ir, rir_rirbox, origin_mesh2ir, origin_rirbox,  _= inference_on_all_models(x_batch, edge_index_batch, batch_indexes,
                                                                            mic_pos.float(), src_pos.float(), 0,
                                                                            mesh2ir, rirbox, DEVICE,
                                                                            SCALE_MESH2IR_BY_ITS_ESTIMATED_STD,
                                                                            SCALE_MESH2IR_GWA_SCALING_COMPENSATION,
                                                                            MESH2IR_USES_LABEL_ORIGIN,
                                                                            RESPATIALIZE_RIRBOX)
                
                # print(rir_mesh2ir.shape, rir_rirbox.shape)
                rirs_mesh2ir.append(rir_mesh2ir[0,:3968])
                rirs_rirbox.append(rir_rirbox[0,:3968])
                origins_mesh2ir.append(origin_mesh2ir)
                origins_rirbox.append(origin_rirbox)
                distances.append(np.linalg.norm(src_pos.squeeze() - mic_pos.squeeze()) / c)
            

            rirs_mesh2ir.extend(rirs_rirbox)
            del rirs_rirbox
            signals_tensor = torch.tensor(signal, device=DEVICE).unsqueeze(0).unsqueeze(1)
            impulse_responses_tensor = torch.nn.utils.rnn.pad_sequence(rirs_mesh2ir, batch_first=True).unsqueeze(1)
            # Assuming impulse responses are all the same length and signals are too, or have been padded accordingly
            results = F.conv1d(signals_tensor, impulse_responses_tensor, padding=0).squeeze().cpu().numpy()
            signal0_mesh2ir = results[0]
            signal1_mesh2ir = results[1]
            signal1_rirbox = results[2]
            signal0_rirbox = results[3]
            # else:
            # signal0_mesh2ir = rirs_mesh2ir[0].cpu().numpy()
            # signal1_mesh2ir = rirs_mesh2ir[1].cpu().numpy()
            # signal0_rirbox = rirs_rirbox[0].cpu().numpy()
            # signal1_rirbox = rirs_rirbox[1].cpu().numpy()

            #############################################
            ################# BEAMFORMING ###############
            #############################################

            nfft = 2048
            hop = 512
            Fmin = 100
            Fmax = 5000

            r = 0 # idx of the reference microphone

            M = mic_pos.shape[-1] # number of microphones
            assert M == len(rirs)
            J = len(rirs[0]) # number of srcs
            assert J == 2

            # THIS ARE THE RIRs
            rir_t = [rirs[i][0] for i in range(len(rirs))]
            rir_i = [rirs[i][1] for i in range(len(rirs))]

            # from list of array to 2D array of the same length
            L = min([len(rir) for rir in rir_t] + [len(rir) for rir in rir_i])
            rir_t = np.stack([rir[:L] for rir in rir_t], axis=0) # [M x L]
            rir_i = np.stack([rir[:L] for rir in rir_i], axis=0) # [M x L]
            print(rir_t.shape, rir_i.shape)

            # ESTIMATE THE RELATIVE TRANSFER FUNCTION
            n = np.random.randn(3*L) # white noise
            hn_t = np.stack([np.convolve(rir_t[i], n) for i in range(M)], axis=0) # convolve with the reference rir
            hn_i = np.stack([np.convolve(rir_i[i], n) for i in range(M)], axis=0) # convolve with the reference rir

            N = librosa.stft(n, n_fft=nfft, hop_length=hop) # [F x T]
            T = N.shape[-1]//4
            HN_t = librosa.stft(hn_t, n_fft=nfft, hop_length=hop) # [M x F x T]
            HN_i = librosa.stft(hn_i, n_fft=nfft, hop_length=hop) # [M x F x T]

            A_t = np.mean((HN_t[:,:,T:2*T] / N[None,:,T:2*T]), axis=-1)
            A_i = np.mean((HN_i[:,:,T:2*T] / N[None,:,T:2*T]), axis=-1)
            print(A_t.shape, A_i.shape)

            rtf_t = A_t / A_t[r] # normalize by the first mic -> we don't need it here, today
            rtf_i = A_i / A_i[r] # normalize by the first mic -> we don't need it here, today

            rtf_t = rtf_t.T # [F x M]
            rtf_i = rtf_i.T # [F x M]

            # Obvervation
            X = librosa.stft(mic_images, n_fft=nfft, hop_length=hop) # [M x F x T]
            X = X.transpose(1,2,0) # [T x F x M]
            F, T, M = X.shape

            print(rtf_i.shape)
            print(rtf_t.shape)
            print(X.shape)

            ## LCMV
            svect = np.stack([rtf_t, rtf_i], axis=-1) # [F x M x J]
            # Noise covariance matrix (interferes is 0 here, we use LCMV!)
            Sigma_n =  np.eye(M)[None,:,:] + 0 * np.einsum('fi,fj-> fij', rtf_i, rtf_i.conj()) # [F x M x M]

            q = np.zeros([J])
            # as default the first src is the tgt
            # the other are considered as interferers
            q[0] = 1

            A = svect       # F x M x J
            Σu = Sigma_n    # F x M x M
            iΣu = np.stack([np.linalg.inv(Σu[f, ...]) for f in range(F)], axis=0) # F x M x M

            num = np.einsum('fiI,fIj-> fij', iΣu, A) # F x I x J
            den = np.einsum('fij,fiI,fiJ->fjJ', A.conj(), iΣu, A) # F x J x J
            den = np.stack([np.linalg.inv(den[f, ...]) for f in range(F)], axis=0) # F x J x J
            W = np.einsum('fij,fjJ,J->fi', num, den, q) # # FxIxJ @ FxJxJ @ Jx1 = Ix1
            Y_tmp = np.einsum('fi,fti-> ft', W.conj(), X) # [F x T]

            # subselct Freqs
            freqs = np.fft.rfftfreq(nfft, 1/fs)
            fidx = np.arange(np.where(freqs > Fmin)[0][0], np.where(freqs < Fmax)[0][-1]) # type:ignore

            Y = np.zeros_like(Y_tmp)
            Y[fidx] = Y_tmp[fidx]

            print(Y.shape)
            librosa.display.specshow(librosa.amplitude_to_db(np.abs(Y), ref=np.max), x_axis='time', y_axis='log', sr=fs, hop_length=hop)



            
            iterations +=1
            if iterations == validation_iterations:
                break
    
    my_list = np.array(my_list)
    df = pd.DataFrame(my_list, columns=["mse_mesh2ir", "mse_rirbox", "mse_origins_mesh2ir", "mse_origins_rirbox"])

    save_path = "./validation/results_sss/" + config['SAVE_PATH'].split("/")[-2] + "/" + config['SAVE_PATH'].split("/")[-1].split(".")[0] + ".csv"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    df.to_csv(save_path)

    print("Validation results saved at: ", save_path)
