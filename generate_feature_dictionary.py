import os
import sys
import argparse
import numpy as np
import cv2

def main():
    # 引数をパースする
    parser = argparse.ArgumentParser("generate face feature dictionary from an face image")
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
    # 画像を開く
    image = cv2.imread(image_path)
    directory = os.path.dirname(image_path)

    if image is None:
        exit()

    # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # モデルを読み込む
    weights = os.path.join('C:/github/face-recognition-opencv/models', "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 特徴を抽出する
    face_feature = face_recognizer.feature(image)
    print(face_feature)
    print(type(face_feature))

    # 特徴を保存する
    basename = os.path.splitext(os.path.basename(image_path))[0]
    dictionary = os.path.join('C:/github/face-recognition-opencv/data/4.feature', basename)
    np.save(dictionary, face_feature)

if __name__ == '__main__':
    main()