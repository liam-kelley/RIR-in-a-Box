from json import load

def load_config_from_file(args):
    with open(args.config_file, 'r') as file:
        config = load(file)
    for key, value in config.items():
        setattr(args, key, value)
    return args
