# Model Configs

Heres a few tips and tricks with model configs as of 2024/03/08.

Activating SYNC_IR_ONSET_IN_LOSSES also activates the start_from_ir_onset option in the Shoebox2RIR. So, if this is activated, you might be able to use shorter RIR lengths, but only a little shorter...
Otherwise, keep RIR lengths to 3968 samples.

If you use ISM_MAX_ORDER = 10, the late energies might be a bit lacking, but you will be able to train on much larger batch sizes!

## Ablation 6

Here's what I want to test in ablation 6.

- [ ] Train dataset = "dp", "nonzero"

FAST TRAINING:

- [ ] ISM_MAX_ORDER = 10 but with larger batch size: This is dumb! I'm limited by IO speed.
- [ ] ISM_MAX_ORDER = 15 but with BATCH_SIZE = 8.

- [ ] RIRBOX_MODEL_ARCHITECTURE = [2 , 3]

- [ ] SYNC_IR_ONSET_IN_LOSSES = [False , True]

- [ ] MRSTFT_HI_Q_TEMPORAL = [False, True]

- [ ] EDC_LOSS_WEIGHT = [0.0, 0.1]
