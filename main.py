import cv2, os, glob
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
import ffmpeg

def print_version():
    print("OpenCV version: {}".format(cv2.__version__))

def configure_cascade():
    cascade_path = ".\haarcascades\haarcascade_frontalface_default.xml"
    assert os.path.isfile(cascade_path)
    return cv2.CascadeClassifier(cascade_path)

def detect_faces_for_file(input_file, output_file, cascade):
    image = cv2.imread(input_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(150, 150))
    white = (255, 255, 255)
    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), white, thickness=2)
        cv2.imwrite(output_file, image)

def detect_faces_for_directory(input_dir, output_dir):
    input_files = glob.glob(os.path.join(input_dir, "*"))
    for input_file in input_files:
        basename = os.path.basename(input_file)
        if basename.endswith(".jpg"):
            output_file = os.path.join(output_dir, basename)
            print("Detecting faces in {}".format(input_file))
            print("Writing to {}".format(output_file))
            detect_faces_for_file(input_file, output_file, configure_cascade())

def scene_detect(input_dir, output_dir):
    input_files = glob.glob(os.path.join(input_dir, "*"))
    for input_file in input_files:
        print("Detecting scenes in {}".format(input_file))
        basename = os.path.basename(input_file)
        if basename.endswith(".mp4"):
            print("Detecting scenes in {}".format(input_file))
            video_manager = VideoManager([input_file])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=30.0))
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            i = 0
            for scene in scene_list:
                print("Scene {}: {}".format(scene[0].get_seconds(), scene[1].get_seconds()))
                basename = os.path.basename(input_file)
                output_file = os.path.join(output_dir, basename + "_" + str(i) + ".jpg")
                print("Writing to {}".format(output_file))
                ffmpeg.input(input_file, ss=scene[0].get_seconds()).output(output_file, vframes=1).run(overwrite_output=True)
                i += 1

def main():
    input_video_dir = ".\\data\\1.input_video"
    snapshot_dir = ".\\data\\2.snapshot"
    facedetect_dir = ".\\data\\3.facedetect"

    scene_detect(input_video_dir, snapshot_dir)
    detect_faces_for_directory(snapshot_dir, facedetect_dir)

if __name__ == "__main__":
    print_version()
    main()
