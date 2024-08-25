import argparse
import json
import os
import tqdm
from utils_io import load_yaml, save_yaml
import subprocess
import shlex


def run_experiments(args):
    log_dir = args.log_dir
    config_path = args.config_path
    exp_file = args.exp_file

    # Load base config
    config = load_yaml(config_path)

    # Load experiments params
    exp_json = json.load(open(exp_file))

    # Create log dir
    os.makedirs(log_dir, exist_ok=True)

    # Setup experiment
    values_for_param = exp_json["param_values"]

    print(f"[ INFO ] Reunning experiments for {exp_json['target_param']}")
    print(f"[ INFO ] Values: {values_for_param}")

    for value in tqdm.tqdm(values_for_param):
        run_name = f"{exp_json['target_param']}_{value}"
        run_log_dir = f"{log_dir}/{run_name}"

        # Overwrite config file
        keys = exp_json["path_on_config"]
        config[keys[0]][keys[1]] = value
        save_yaml(config_path, config)

        # Run training
        subprocess.call(
            shlex.split(
                f"./src/scripts/train_model.sh {run_log_dir} {exp_json['exp_name']} {run_name}"
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--log-dir",
        help="Folder where the logs will be saves",
        dest="log_dir",
        default="exp_logs",
        required=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Path to base config file",
        dest="config_path",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--exp-file",
        help="Path to the json experiment files describing the experiments variables",
        dest="exp_file",
        required=True,
    )

    args = parser.parse_args()

    run_experiments(args)
