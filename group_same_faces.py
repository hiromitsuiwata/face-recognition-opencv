import os
import sys
import glob
import numpy as np
import cv2
import argparse

COSINE_THRESHOLD = 0.363
NORML2_THRESHOLD = 1.128

# 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
def match(recognizer, feature1, dictionary):
    for element in dictionary:
        user_id, feature2 = element
        score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        if score > COSINE_THRESHOLD:
            return True, (user_id, score)
    return False, ("", 0.0)

def main():
    # 引数をパースする
    parser = argparse.ArgumentParser('generate aligned face images from an image')
    parser.add_argument('feature_dir', help='input feature file dir path')
    args = parser.parse_args()
    feature_dir = args.feature_dir

    # 辞書を読み込む
    dictionary = []
    files = glob.glob(os.path.join(feature_dir, "*.npy"))
    for file in files:
        feature = np.load(file)
        file_name = os.path.splitext(os.path.basename(file))[0]
        dictionary.append((file_name, feature))

    # モデルを読み込む
    weights = os.path.join('C:/github/face-recognition-opencv/models', "face_recognizer_fast.onnx")
    face_recognizer = cv2.FaceRecognizerSF_create(weights, "")

    # 特徴を比較する
    group = []# 重複しているファイルを格納するリスト

    for file_a in files:
        file_name_a = os.path.splitext(os.path.basename(file_a))[0]
        feature_a = np.load(os.path.join(feature_dir, file_a))
        # グループが空の場合は追加する
        if len(group) == 0:
            group.append([file_name_a])
            continue

        # すでにグループに登録されているものと一致しているか比較する
        for g in group:
            for file in g:
                if file == file_name_a:
                    continue
                feature = np.load(os.path.join(feature_dir, file + ".npy"))
                score = face_recognizer.match(feature, feature_a, cv2.FaceRecognizerSF_FR_COSINE)
                if score > COSINE_THRESHOLD:
                    g.append(file_name_a)
                    break
            else:
                continue
            break
        else:
            group.append([file_name_a])
        
    for g in group:
        print(g)

if __name__ == '__main__':
    main()