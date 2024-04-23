# Losses

- ShoeboxLosses.py : This script contains the losses that depend on the latent shoebox representation. Namely:
  - SBoxRoomDimensionsLoss
  - SBoxAbsorptionLoss
  - MicSrcConfigurationLoss
- RIRLosses.py : This script contains the losses that depend on the Room Impulse Response. Namely:
  - EnergyDecay_Loss
  - MRSTFT_Loss
  - AcousticianMetrics_Loss
    - These are all done together for optimizations (D, C80, DRR, RT60)

All losses now have easily toggleable options:

- frequency_wise (edecay, mrstft)
- synchronize_TOA (edecay, mrstft, acoustician)
- normalize_TOA (edecay, mrstft, acoustician)
- normalize_decay_curve (edecay)
- deemphasize_early_reflections (edecay, mrstft)
- Normalize total energy (Acoustician)
- pad_to_same_length (edecay, mrstft, acoustician)
- crop_to_same_length (edecay, mrstft, acoustician) # this is preferred, see AV-RIR's paper.
