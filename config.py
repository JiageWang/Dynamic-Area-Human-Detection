# 相机参数设置
CAP_PATH = 0
CAP_WIDTH = 640
CAP_HEIGHT = 480

# 初始化位置区域
INIT_ROI = [[370, 190], [370, 290], [270, 290], [270, 190]]

# 模型、配置路径
OPENPOSE_PATH = "checkpoints/checkpoint_iter_370000.pth"

# Openpose配置
OPENPOSE_STRIDE = 8
OPENPOSE_SMOOTH = 1
OPENPOSE_HEIGHT_SIZE = 256
OPENPOSE_UPSAMPLE_RATIO = 4
OPENPOSE_IMG_SCALE = 1 / 256
OPENPOSE_PAD_VALUE = (0, 0, 0)
OPENPOSE_IMG_MEAN = (128, 128, 128)
OPENPOSE_THRESHOLD = 5