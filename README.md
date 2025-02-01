# face-recognition-opencv

- https://qiita.com/UnaNancyOwen/items/8c65a976b0da2a558f06#%E3%81%8A%E3%81%BE%E3%81%91opencv%E3%81%AE%E5%AE%9F%E8%A3%85%E3%81%AE%E6%B0%97%E3%81%AB%E5%85%A5%E3%82%89%E3%81%AA%E3%81%84%E3%81%A8%E3%81%93%E3%82%8D
- https://www.scenedetect.com/docs/latest/api/scene_manager.html#
- https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- https://drive.google.com/file/d/1ClK9WiB492c5OZFKveF3XiHCejoOxINW/view

```pwsh
python .\save_scenes.py c:\github\face-recognition-opencv\data\1.input_video\
python .\generate_aligned_faces.py C:\github\face-recognition-opencv\data\2.snapshot\
python .\generate_feature_dictionary.py c:\github\face-recognition-opencv\data\3.facedetect
python .\group_same_faces.py C:\github\face-recognition-opencv\data\4.feature
```
