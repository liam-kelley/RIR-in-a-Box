from json import load

def load_config_from_file(args):
    with open(args.config_file, 'r') as file:
        config = load(file)
    for key, value in config.items():
        setattr(args, key, value)
    return args

# parser = argparse.ArgumentParser()
# parser.add_argument('--RIRBOX_MODEL_ARCHITECTURE',  type=int,   default=2)
# parser.add_argument('--PRETRAINED_MESHNET',         type=bool,  default=True)
# parser.add_argument('--TRAIN_MESHNET',              type=bool,  default=False)
# parser.add_argument('--LEARNING_RATE',              type=float, default=1e-4)
# parser.add_argument('--EPOCHS',                     type=int,   default=4)
# parser.add_argument('--BATCH_SIZE',                 type=int,   default=4)
# parser.add_argument('--DEVICE',                     type=str,   default='cuda')
# parser.add_argument('--ISM_MAX_ORDER',              type=int,   default=10, help='This value is O^3 in memory. 10 should be minimum, 15 is recommended.' )
# parser.add_argument('--do_wandb',                   type=bool,  default=False)
# parser.add_argument('--config_file',                type=str,   default=None, help='Path to configuration file.')
# args, _ = parser.parse_known_args()
# if args.config_file: args = load_config_from_file(args)
# RIRBOX_MODEL_ARCHITECTURE = args.RIRBOX_MODEL_ARCHITECTURE
# PRETRAINED_MESHNET = args.PRETRAINED_MESHNET
# TRAIN_MESHNET = args.TRAIN_MESHNET
# LEARNING_RATE = args.LEARNING_RATE
# EPOCHS = args.EPOCHS
# BATCH_SIZE = args.BATCH_SIZE
# DEVICE = args.DEVICE
# ISM_MAX_ORDER = args.ISM_MAX_ORDER
# do_wandb = args.do_wandb
