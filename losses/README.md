# Losses

This folder includes implementations of various losses, implemented as torch.nn.modules, for RIRs and for RIRBox's latent shoebox representation.

- ShoeboxLosses.py : This script contains the losses that depend on the latent shoebox representation. Namely:
  - SBoxRoomDimensionsLoss
  - SBoxAbsorptionLoss
  - MicSrcConfigurationLoss
- RIRLosses.py : This script contains the losses that depend on the Room Impulse Response. Namely:
  - EnergyDecay_Loss : MSE between target and estimated EDC or EDR.
  - MRSTFT_Loss : MRSTFT between target and estimated.
  - AcousticianMetrics_Loss : MSE between target and estimated D, C80, DRR, and RT60.
    - These are all done together for optimization.
  - RIR_MSE_Loss : MSE between target and estimated RIR
  - DRR_Loss : MSE between target and estimated Direct Reverberant Ratios (DRR)
