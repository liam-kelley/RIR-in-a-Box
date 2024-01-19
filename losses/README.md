# Losses

- ShoeboxLosses.py : This script contains the losses that depend on the latent shoebox representation. Namely:
  - SBoxRoomDimensionsLoss
  - SBoxAbsorptionLoss
  - MicSrcConfigurationLoss
- RIRLosses.py : This script contains the losses that depend on the Room Impulse Response. Namely:
  - EnergyDecay_Loss
  - EnergyBins_Loss (New)
  - MRSTFT_Loss
  - AcousticianMetrics_Loss
    - These are all done together for optimizations
D
C80
RT60 (new implementation with mean around median)
RT60 betas (new)

All losses now have easily toggleable options:

- frequency_wise (edecay, ebins, mrstft, acoustician)
- synchronize_dp (edecay, ebins, mrstft, acoustician)
- normalize_dp (edecay, ebins, mrstft, acoustician)
- normalize_decay_curve (edecay)
- deemphasize_early_reflections (edecay, ebins, mrstft)
- Normalize total energy (Acoustician)
- pad_to_same_length (edecay, ebins, mrstft, acoustician)
- crop_to_same_length (edecay, ebins, mrstft, acoustician) # this is preferred, see AV-RIR's paper.
