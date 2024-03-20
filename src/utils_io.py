import cv2
import json
import os
import glob
import json
import yaml


def save_config(config, output_path):
    """
    Saves the config dict into a Json file

    Params:
       config -> Dict: Dict of training and model configs

       output_path -> Str: Output path of the json file

    Return:
       json_path -> Str: Path to the file created
    """

    # Make output dir if necessary
    os.makedirs(output_path, exist_ok=True)
    json_path = f"{output_path}/config.json"

    with open(json_path, "w+") as f:
        json.dump(config, fp=f, indent=2)

    return json_path


def make_video(frames, output_folder, fps=30, color=False):
    """
    Recieves a list of frames and make a video out of it

    Args:
        Framse -> List[np.array()]: List of frames. All frames must be of the same size

        output_folder -> Srt: Output folder where the video will be created

        fps -> Int: Number of frames per second

        color -> Boolean: True if frames are RGB, defaults to False (gray scale)

    """
    frame_size = frames[0].shape[:2]

    video_writer = cv2.VideoWriter(
        f"{output_folder}/video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
        isColor=color,
    )

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()


def load_json(file):
    """
    Load a json file from path

    Args:
        file -> str: Path to the json file

    Return:
        json_file -> dict: Json file loaded


    """
    with open(file, "r") as f:
        json_file = json.load(f)

    return json_file


def load_yaml(file):
    """
    Load a yaml file from path

    Args:
        file -> str: Path to the yaml file

    Return:
        yaml_file -> dict: Yaml file loaded


    """

    with open(file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def save_yaml(file, yaml_obj):
    """
    Save a yaml file from path

    Args:
        file -> str: Path to the yaml file

        yaml_obj -> dict: Dict with the yaml content

    Return:
        success -> boll: Flag indicating success or failure
    """

    try:
        yaml.dump(yaml_obj, open(file, "w"), indent=2)
    except:
        print("Failed to save the YAML file")
        return False

    return True


def save_eval(metrics, log_dir):
    pass


if __name__ == "__main__":
    frames = glob.glob("logs/BetaVAE/version_0/Reconstructions/*")

    frames = [cv2.imread(frame, cv2.IMREAD_GRAYSCALE) for frame in frames]
    make_video(frames, "./", fps=5)
