from openni import openni2
import numpy as np
import cv2


def mousecallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(y, x, dpt[y, x])


if __name__ == "__main__":
    openni2.initialize()
    dev = openni2.Device.open_any()
    print(dev.get_device_info())
    depth_stream = dev.create_depth_stream()
    dev.set_image_registration_mode(True)
    depth_stream.start()
    cap = cv2.VideoCapture(1)
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth', mousecallback)
    while True:

        frame = depth_stream.read_frame()
        # 转换数据格式
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')

        dpt2 *= 255
        # 对于为什么要乘于255的解答
        # 深度图像的深度值 是按照16位长度（两字节）的数据格式存储的，也可以认为前八位是高字节，后八位是低字节。
        # 因此一张深度图像如果是 640480分辨率的话，那么图像字节大小 就是 640480*2，其中一个字节是8位（255）
        dpt = dpt1 + dpt2
        # cv2里面的函数，就是类似于一种筛选
        '假设我们需要让我们的深度摄像头感兴趣的距离范围有差别地显示，那么我们就需要确定一个合适的alpha值，公式为：有效距离*alpha=255，' \
        '假设我们想让深度摄像头8m距离内的深度被显示，>8m的与8m的颜色显示相同，那么alpha=255/(8*10^3)≈0.03，' \
        '假设我们想让深度摄像头6m距离内的深度被显示，>6m的与6m的颜色显示相同，那么alpha=255/(6*10^3)≈0.0425'
        dim_gray = cv2.convertScaleAbs(dpt, alpha=0.17)
        # 对深度图像进行一种图像的渲染，目前有11种渲染方式，大家可以逐一去试下
        depth_colormap = cv2.applyColorMap(dim_gray, 2)  # 有0~11种渲染的模式

        cv2.imshow('depth', depth_colormap)

        ret, frame = cap.read()
        cv2.imshow('color', frame)

        key = cv2.waitKey(1)
        if int(key) == ord('q'):
            break

    depth_stream.stop()
    dev.close()