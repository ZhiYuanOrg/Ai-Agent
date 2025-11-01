import argparse
import cv2
import numpy as np
import onnxruntime as ort
from utils.read_cls import load_class_names
from openni import openni2
# 类外定义类别映射关系，使用字典格式
CLASS_NAMES =load_class_names("models/coco.yaml")

class Detector:
    """目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model,  confidence_thres=0.5, iou_thres=0.5, use_depth_camera=False):
        """
        初始化 Detector 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.use_depth_camera = use_depth_camera
        # 加载类别名称
        self.classes = CLASS_NAMES

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 深度相机相关变量
        self.dev = None
        self.depth_stream = None
        self.cap = None
        self.dpt = None

        # 初始化深度相机（如果启用）
        if self.use_depth_camera:
            self.initialize_depth_camera()

    def initialize_depth_camera(self):
        """初始化深度相机"""
        try:
            openni2.initialize()
            self.dev = openni2.Device.open_any()
            print("深度相机设备信息:", self.dev.get_device_info())

            # 创建深度流
            self.depth_stream = self.dev.create_depth_stream()
            self.dev.set_image_registration_mode(True)
            self.depth_stream.start()
            print("深度流已启动")
            # 创建彩色摄像头流
            self.cap = cv2.VideoCapture(1)

            # 创建深度图像显示窗口并设置鼠标回调
            cv2.namedWindow('depth')
            cv2.setMouseCallback('depth', self.mouse_callback)

            print("深度相机初始化成功")
        except Exception as e:
            print(f"深度相机初始化失败: {e}")
            self.use_depth_camera = False

    def mouse_callback(self, event, x, y, flags, param):
        """深度图像鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDBLCLK and self.dpt is not None:
            if y < self.dpt.shape[0] and x < self.dpt.shape[1]:
                print(f"坐标({y}, {x})的深度值: {self.dpt[y, x]}")

    def get_depth_frame(self):
        """获取深度图像帧"""
        if not self.use_depth_camera or self.depth_stream is None:
            return None

        try:
            frame = self.depth_stream.read_frame()
            # 转换数据格式
            dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
            dpt2 *= 255
            self.dpt = dpt1 + dpt2

            # 转换为可显示的深度图像
            dim_gray = cv2.convertScaleAbs(self.dpt, alpha=0.17)
            depth_colormap = cv2.applyColorMap(dim_gray, 2)
            depth_colormap =cv2.flip(depth_colormap, 1)
            return depth_colormap
        except Exception as e:
            print(f"获取深度帧失败: {e}")
            return None

    def get_color_frame(self):
        """获取彩色图像帧"""
        if self.use_depth_camera and self.cap is not None:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def preprocess(self, img):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = img
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0

        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸

        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 为图像添加边框以达到目标尺寸
        # 先计算一侧的填充，然后另一侧用剩余部分
        top = int(round(dh / 2))
        bottom = int(dh - top)  # 确保 top + bottom = dh

        left = int(round(dw / 2))
        right = int(dw - left)  # 确保 left + right = dw
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        print(f"Final letterboxed image shape: {img.shape}")

        return img, (r, r), (dw, dh)

    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw/2  # 移除填充
                y -= self.dh/2
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
        return input_image

    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。

        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def initialize_model(self):
        """初始化ONNX模型"""
        # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
        session = ort.InferenceSession(
            self.onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else [
                "CPUExecutionProvider"],
        )

        # 获取模型的输入形状
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")

        return session

    def process_image(self, session, img):
        """处理单张图像"""
        # 预处理图像数据
        img_data = self.preprocess(img)

        # 运行推理
        outputs = session.run(None, {session.get_inputs()[0].name: img_data})

        # 后处理
        result_img = self.postprocess(img.copy(), outputs)
        return result_img

    def run_realtime(self):
        """实时运行目标检测（使用深度相机或普通摄像头）"""
        session = self.initialize_model()

        if not self.use_depth_camera:
            print("正在使用普通摄像头...")
            # 使用普通摄像头
            self.cap = cv2.VideoCapture(0)

        print("开始实时检测，按'q'退出...")

        while True:
            # 获取彩色帧
            color_frame = self.get_color_frame()
            if color_frame is None:
                print("无法获取彩色帧")
                break

            # 进行目标检测
            detected_frame = self.process_image(session, color_frame)
            cv2.imshow('Object Detection', detected_frame)

            # 如果使用深度相机，显示深度图像
            if self.use_depth_camera:
                depth_frame = self.get_depth_frame()
                if depth_frame is not None:
                    cv2.imshow('depth', depth_frame)

            # 退出条件
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # 释放资源
        self.release_resources()

    def release_resources(self):
        """释放所有资源"""
        if self.cap is not None:
            self.cap.release()
        if self.depth_stream is not None:
            self.depth_stream.stop()
        if self.dev is not None:
            self.dev.close()
        cv2.destroyAllWindows()

    def process_single_image(self, image_path):
        """处理单张图像（原有功能）"""
        session = self.initialize_model()

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像: {image_path}")
            return None

        # 处理图像
        result_img = self.process_image(session, img)

        # 显示结果
        cv2.imshow('Object Detection', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return result_img


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/yolo11n.onnx', help='ONNX模型路径')
    parser.add_argument('--image', type=str, default=None, help='输入图像路径（如果为None则使用摄像头）')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU阈值')
    parser.add_argument('--depth-camera', type=bool, default=True, help='使用深度相机')
    args = parser.parse_args()

    # 创建检测器实例
    detector = Detector(
        onnx_model=args.model,
        confidence_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        use_depth_camera=args.depth_camera
    )

    # 运行检测
    if args.image:
        # 处理单张图像
        detector.process_single_image(args.image)
    else:
        # 实时检测
        detector.run_realtime()


