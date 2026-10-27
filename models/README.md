# Models

`face_detection_yunet_2023mar.onnx` is the YuNet face detector from the OpenCV Zoo (github.com/opencv/opencv_zoo, `models/face_detection_yunet`). It is used only to blur faces in uploaded photos before anything else happens, and it runs on OpenCV 5's DNN engine via `cv2.FaceDetectorYN`.

YuNet is published in the OpenCV Zoo under the MIT license. Check the upstream LICENSE file before redistributing it.

ARGUS uses no other learned model: gauge reading is geometric computer vision.
