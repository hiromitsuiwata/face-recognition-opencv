import scenedetect
import argparse
import os

def main():
    # 引数をパースする
    parser = argparse.ArgumentParser('save scenes')
    parser.add_argument('movie_file_dir')
    args = parser.parse_args()

    movie_file_dir = args.movie_file_dir
    print(movie_file_dir)
    files = os.listdir(movie_file_dir)
    for file in files:
        movie_filepath = os.path.join(movie_file_dir, file)
        if os.path.isfile(movie_filepath):
            save_scenes(movie_filepath)

def save_scenes(video_path):
    print(video_path)
    video = scenedetect.open_video(video_path)
    scene_manager = scenedetect.SceneManager()
    scene_manager.add_detector(scenedetect.ContentDetector())

    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()

    scenedetect.save_images(scenes, video, num_images=1, image_extension="jpg", output_dir="./data/2.snapshot")

main()