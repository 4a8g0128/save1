#!/usr/bin/env python

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import cv2
import csv
import sys
import os
import math
import time 
import numpy as np
import mediapipe as mp
import qtmodern.styles
import qtmodern.windows

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding
    
    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).
    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)
    
    return landmarks

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.
    
    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

class creatcsv():  

   def __init__(self ,landmarks ,csvs_out_folder,pose_name) :
        self.landmarks = landmarks
        self.csvs_out_folder = csvs_out_folder
        self.pose_name=pose_name
        
   def creat(self) :
        
       # self.pose_class_names = sorted([n for n in os.listdir(self._images_in_folder) if not n.startswith('.')])

        if not os.path.exists(self.csvs_out_folder):
            os.makedirs(self.csvs_out_folder)

       # for pose_class_name in self.pose_class_names:
           # print('Bootstrapping ', pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            #csv_out_path = os.path.join(self.csvs_out_folder, pose_class_name + '.csv')
        csv_out_path = os.path.join(self.csvs_out_folder, "pose"+ '.csv')
        with open(csv_out_path, 'w') as csv_out_file:
            csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csv_out_writer.writerow([self.pose_name] + self.landmarks.flatten().astype(np.str).tolist())

class PoseSample(object):

  def __init__(self, name, landmarks, class_name, embedding):
    self.name = name
    self.landmarks = landmarks
    self.class_name = class_name
    
    self.embedding = embedding

class PoseSampleOutlier(object):

  def __init__(self, sample, detected_class, all_classes):
    self.sample = sample
    self.detected_class = detected_class
    self.all_classes = all_classes
      
class poencsv(object):
    def __init__(self,pose_samples_folder,
                fullbody,
                file_extension='csv',
                file_separator=',',
                n_landmarks=33,
                n_dimensions=3,
                top_n_by_max_distance=30,
                top_n_by_mean_distance=10, 
                axes_weights=(1., 1., 0.2)):
        self.full_boby = fullbody
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights
        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                 file_extension,
                                                 file_separator,
                                                 n_landmarks,
                                                 n_dimensions,
                                                 fullbody)
    def _load_pose_samples(self,
                         pose_samples_folder,
                         file_extension,
                         file_separator,
                         n_landmarks,
                         n_dimensions,
                         fullbody):
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]
      
        # Parse CSV.
        with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=file_separator)
            for row in csv_reader:
                assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                pose_samples.append(PoseSample(
                    name=row[0],
                    landmarks=landmarks,
                    class_name=class_name,
                    embedding=fullbody(landmarks),
          ))

        return pose_samples                                                 
    def __cell__(self, pose_landmarks):
        """Classifies given pose.
        Classification is done in two stages:
        * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
        * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.
        
        Args:
        pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).
        Returns:
        Dictionary with count of nearest pose samples from the database. Sample:
            {
            'pushups_down': 8,
            'pushups_up': 2,
            }
        """
           

        # Get given pose embedding.
        pose_landmarks = self.full_boby(pose_landmarks)
        flipped_pose_embedding = self.full_boby(pose_landmarks * np.array([-1, 1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        #max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
            np.max(np.abs(sample.embedding - pose_landmarks) * self._axes_weights),
            np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
        )

        mean_dist = 100 -  round(max_dist)

        return mean_dist
  

class MainApp(QWidget):
    #pose_landmarks檔案路徑
   # csvinputfolder='Users/xushuyu/Desktop/AIRehabilitation-main'

    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(320, 240)
        self.setup_ui()
        self.setup_camera()
        self.fullbody =  FullBodyPoseEmbedder()
        self.pose_open=poencsv(pose_samples_folder='/Users/xushuyu/Desktop/ARTS-3bd075fb70a33877a95eb5a95226871a9f237316/pos_csvs',
                fullbody=self.fullbody)
        t=time.time()
        t2=time.localtime(t)
        self.t3=time.mktime(t2)
        self.t4=self.t3+30
               
                    

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        #self.image_label.setFixedSize(self.video_size)
        self.image_label.setFixedSize(QSize(640, 480))

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.、、
        """
        self.capture = cv2.VideoCapture(0)
        if not  self.capture.isOpened():
            print("Cannot open camera")
            exit()
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)
    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
    
        # 啟用姿勢偵測
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            ret, img = self.capture.read()
            t=time.time()
            t2=time.localtime(t)
            t3=time.mktime(t2)
            x=0
            second=round(self.t4-t3)
            if second is x :
                print('Finish')
                exit()

            if not ret:
                print("Cannot receive frame")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = pose.process(img)                  # 取得姿勢偵測結果
            # 根據姿勢偵測結果，標記身體節點和骨架
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            img = cv2.flip(img, 1)
            #把影像調整到標簽QLabel一樣大
            image = QImage(img, img.shape[1], img.shape[0],  img.strides[0], QImage.Format_RGB888)
            myScaledPixmap = QPixmap.fromImage(image).scaled( self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(myScaledPixmap)
            
            # 取得landmarks
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                frame_height, frame_width = img.shape[0], img.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                        for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
           
            #紀錄User_landmarks
            csvinputfolder ='landmarks_Save'
            posename='123'
            csv=creatcsv(landmarks = pose_landmarks ,csvs_out_folder=csvinputfolder,pose_name=posename)
            csv.creat()
               
            poselandmarks=self.pose_open.__cell__(pose_landmarks)
            #349 350畫面輸出 260訓練資料位置 340姿勢名稱
            print(second)
            print(str(poselandmarks)+'%')


           


           
       
    
        






if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    # code below界面更好看, 但是速度會變慢
    # qtmodern.styles.dark(app)
    # mw = qtmodern.windows.ModernWindow(win)
    # mw.show()
    win.show()
    sys.exit(app.exec_())