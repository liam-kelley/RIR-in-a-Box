# Model Configs

Heres a few tips and tricks with model configs as of 2024/03/08.

Activating SYNC_IR_ONSET_IN_LOSSES also activates the start_from_ir_onset option in the Shoebox2RIR. So, if this is activated, you might be able to use shorter RIR lengths, but only a little shorter...
Otherwise, keep RIR lengths to 3968 samples.

If you use ISM_MAX_ORDER = 10, the late energies might be a bit lacking, but you will be able to train on much larger batch sizes!

## Ablation 6

Here's what I want to test in ablation 6.

- [ ] Train dataset = ["dp", "nonzero"]

- [ ] [ISM_MAX_ORDER = 10, BATCH_SIZE = 28, RIR_LENGTH = 3000] / [ISM_MAX_ORDER = 15, BATCH_SIZE = 8, RIR_LENGTH = 3600]

- [ ] RIRBOX_MODEL_ARCHITECTURE = [2 , 3]

- [ ] SYNC_IR_ONSET_IN_LOSSES = [False , True]
- [ ] MRSTFT_HI_Q_TEMPORAL = [False, True]
- [ ] NORMALIZE_BY_DIST_IN_LOSSES = [False, True]

- [ ] EDC_LOSS_WEIGHT = [0.0, 0.5]
- [ ] RT60_LOSS_WEIGHT = [0.0, 200]
- [ ] MIC_SRC_DISTANCE_LOSS_WEIGHT = [0.0, 2.0]
