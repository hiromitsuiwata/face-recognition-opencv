import os
import argparse
import numpy as np
import cv2

def main():
    # 引数をパースする
    parser = argparse.ArgumentParser('generate aligned face images from an image')
    parser.add_argument('image_dir', help='input image file dir path')
    args = parser.parse_args()

    # 引数から画像ファイルのパスを取得
    print(args.image_dir)
    files = os.listdir(args.image_dir)
    for file in files:
        image_path = os.path.join(args.image_dir, file)
        if os.path.isfile(image_path):
            process_file(image_path)

def process_file(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(image_name)
    directory = os.path.dirname(image_path)

    # 画像を開く
    image = cv2.imread(image_path)
    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    weights = os.path.join('C:/github/face-recognition-opencv/models', 'face_detection_yunet_2023mar.onnx')
    face_detector = cv2.FaceDetectorYN_create(weights, '', (0, 0))
    weights = os.path.join('C:/github/face-recognition-opencv/models', 'face_recognizer_fast.onnx')
    face_recognizer = cv2.FaceRecognizerSF_create(weights, '')

    # 入力サイズを指定する
    height, width, _ = image.shape
    face_detector.setInputSize((width, height))

    # 顔を検出する
    _, faces = face_detector.detect(image)

    # 検出された顔を切り抜く
    aligned_faces = []
    if faces is not None:
        for face in faces:
            aligned_face = face_recognizer.alignCrop(image, face)
            aligned_faces.append(aligned_face)

    # 画像を表示、保存する
    for i, aligned_face in enumerate(aligned_faces):
        cv2.imwrite(os.path.join('C:/github/face-recognition-opencv/data/3.facedetect', image_name + '_face{:03}.jpg'.format(i + 1)), aligned_face)

if __name__ == '__main__':
    main()