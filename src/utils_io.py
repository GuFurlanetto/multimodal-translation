import cv2
import json
import os


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

        color -> Boolean:True if frames are RGB, defaults to False (gray scale)

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


if __name__ == "__main__":
    config = {"Nome": "Gustavo", "Idade": 22}

    save_config(config, "teste_folder")
