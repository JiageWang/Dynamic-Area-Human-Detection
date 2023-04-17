import sys
from multiprocessing import Queue

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QApplication

from config import *
from openpose_light import OpenposeLight

pg.setConfigOptions(imageAxisOrder='row-major')


class MyApp(pg.GraphicsLayoutWidget):
    q_imgs = Queue(maxsize=1)
    q_lines = Queue(maxsize=1)
    q_poses = Queue(maxsize=1)

    def __init__(self):
        super(MyApp, self).__init__(border=True)

        # 组件
        self.pen = QPen(Qt.red, 2, Qt.SolidLine)
        self.timer = QTimer()
        self.cap = cv2.VideoCapture(CAP_PATH)
        self.roi = pg.PolyLineROI(INIT_ROI, closed=True)
        self.image_item = pg.ImageItem()

        self.openpose = OpenposeLight(OPENPOSE_PATH)

        # 初始化
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        self.init_grahics_widget()
        self.roi.setPen(self.pen)
        self.timer.start(10)
        self.setAspectLocked(True)

        # 信号与槽
        self.timer.timeout.connect(self.update_image)

    def init_grahics_widget(self):
        """初始化viewbox添加图像与ROI元素"""
        # 添加标题
        # self.addLabel("智能监控系统", row=0, col=0, size="50px")
        # 添加view box以存放图像与roi
        view_box = self.addViewBox(lockAspect=True, row=1, col=0)
        view_box.addItem(self.roi)
        view_box.addItem(self.image_item)

    def update_image(self):
        """计算器触发更新显示的图像"""
        if not self.cap.isOpened():
            print("camera not accupied")
            return
        ret, img = self.cap.read()
        img = self.forward_models_series(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, -1)
        self.image_item.setImage(img)

    def forward_models_series(self, img):
        """AI模型处理"""
        poses = self.openpose.predict(img)
        dists = self.roi_person_dists(poses)  # 判断每个pose相对roi距离

        img = self.openpose.draw_poses(img, poses, dists)
        # 集成其他模型
        return img

    def match_helmet_person(self, poses, heads):
        '''

        :param poses: pose.head_bbox (x, y, w, h)
        :param heads: [[x1,y1,x2,y2,label,score]]
        :return:
        '''
        heads_copy = heads.copy()
        poses = sorted(poses, key=lambda pose: pose.confidence, reverse=True)
        for pose in poses:
            heads_iou = []
            if not heads_copy: break
            # 计算所有检测结果对当前姿态的iou
            for head in heads_copy:
                pose_head = [
                    pose.head_bbox[0],
                    pose.head_bbox[1],
                    pose.head_bbox[0] + pose.head_bbox[2],
                    pose.head_bbox[1] + pose.head_bbox[3]
                ]
                heads_iou.append(self.iou(pose_head, head[:4]))
            # 匹配最大iou
            index = np.argmax(heads_iou)
            value = np.max(heads_iou)
            if value > MATCH_IOU and heads_copy[index][4] == 'helmet':
                pose.has_helmet = True
                del heads_copy[index]

    @staticmethod
    def iou(box1, box2):
        '''
        两个框（二维）的 iou 计算

        注意：边框以左上为原点

        box:[top, left, bottom, right]
        '''
        in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
        in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
        inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        iou = inter / union
        return iou

    def roi_person_dists(self, poses):
        """计算人与ROI区域的距离"""
        dists = []
        for pose in poses:
            # 根据脚踝位置判断
            l_ank = pose.keypoints[pose.kpt_names.index('l_ank')].astype(int)
            r_ank = pose.keypoints[pose.kpt_names.index('r_ank')].astype(int)
            l_ank = (CAP_WIDTH - l_ank[0], CAP_HEIGHT - l_ank[1])
            r_ank = (CAP_WIDTH - r_ank[0], CAP_HEIGHT - r_ank[1])
            l_dist = cv2.pointPolygonTest(self.contour, l_ank, True) if l_ank[0] != -1 else 0
            r_dist = cv2.pointPolygonTest(self.contour, r_ank, True) if r_ank[0] != -1 else 0
            dists.append(max(l_dist, r_dist))
        return dists

    @property
    def contour(self):
        """获取ROI的边界点"""
        roi_points = []
        for handle in self.roi.handles:
            pos = list(handle['pos'])
            pos[0] += self.roi.pos()[0]
            pos[1] += self.roi.pos()[1]
            roi_points.append(pos)
        return np.array(roi_points, dtype=np.float32)[:, np.newaxis, :]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
