from copy import deepcopy
from pathlib import Path

def save_config(parser, args, config_path):
    """Save the config file (configargparse functionality)."""
    config_args = deepcopy(args)
    for key, value in vars(config_args).items():
        if isinstance(value, Path):
            setattr(config_args, key, str(value))
            
    parser.write_config_file(parsed_namespace=config_args, output_file_paths=[str(config_path)])