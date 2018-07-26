import cv2
import numpy as np
import visdom
import time
import os


class TestViewer():
    """
    test_video : test video 하나의 filename (각 파일명 맨 뒤에 ground true hv의 frame이 적혀있음)
    extracted_hv : test_video 랑 같은 제목, 다른 확장자(npy)를 가지는 filename. numpy array를 가지고 있으며 각 snippet(48fs)마다 0, 1값이 표시됨.
    예상되는 애들은 00000011111111111000뭐 이런식인데[얘는 구현함] 0000011100111111100111이렇게 되는 경우도 생각해보자!!
    """

    def __init__(self, test_video, extracted_hv):

        self.test_video = test_video
        self.extracted_hv = extracted_hv

        # test video를 frame별로 불러와서 numpy array로 test_raw에 저장함.
        cap = cv2.VideoCapture(self.test_video)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                # HWC2CHW
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            else:
                break
        cap.release()

        test_raw = np.concatenate(frames)
        self.test_raw = test_raw.reshape(-1, 3, 270, 480)

    def show(self, item = -1):
        if item ==-1:
            self.showrv()
            self.showthv()
            self.showehv()
        elif item==0:
            self.showrv()
        elif item==1:
            self.showthv()
        elif item==2:
            self.showehv()
        else:
            pass

    def showrv(self):

        viz0 = visdom.Visdom(use_incoming_socket=False)

        for f in range(0, self.test_raw.shape[0]):
            viz0.image(self.test_raw[f, :, :, :], win="gt video", opts={'title': 'TEST_RAW'}, )
            time.sleep(0.01)

    def showthv(self):
        viz1 = visdom.Visdom(use_incoming_socket=False)
        # 이 과정은 test_true_hv를 보여주기 위해 test_raw에서 hv frame을 index함,
        filename = os.path.split(self.test_video)[-1]

        h_start = filename.index("(")
        h_end = filename.index(")")

        h_frames = filename[h_start + 1: h_end]
        # h_frames = "42, 120" or "nohv"

        if "," in h_frames:
            s, e = h_frames.split(',')
            h_start, h_end = int(s), int(e)

        else:
            h_start, h_end = 0, 0
        for f in range(h_start, h_end):
            if (h_start == h_end):
                # no highlight라고 얘기하고 visdom에다가 싶은데?
                break

            viz1.image(self.test_raw[f, :, :, :], win="gt1 video", opts={'title': 'TEST_TRUE_HV'}, )
            time.sleep(0.01)

    def showehv(self):
        viz2 = visdom.Visdom(use_incoming_socket=False)
        # 이 과정은 test_extracted_hv를 보여주기 위해 test_raw에서 hv frame을 index함.
        ext = np.load(self.extracted_hv)
        ext_idx = np.asarray(ext.nonzero()).squeeze()
        print(ext_idx[0], ext_idx[-1])

        if ext_idx == []:
            e_start, e_end = 0, 0
        else:
            e_start = ext_idx[0] * 6
            e_end = ext_idx[-1] * 6 + 48
            # "42, 120" 이라면 "7, 12"

        for f in range(e_start, e_end):
            if (e_start == e_end):
                # no highlight라고 얘기하고 visdom에다가 싶은데?
                break

            viz2.image(self.test_raw[f, :, :, :], win="gt2 video", opts={'title': 'TEST_Extracted_HV'}, )
            time.sleep(0.01)


if __name__ == "__main__":
    test_video = r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\testRV\testRV00(42,120).mp4"
    extracted_hv = r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\testRV\testRV00(42,120).npy"

    cap = cv2.VideoCapture(test_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            # HWC2CHW
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
        else:
            break
    cap.release()

    size = int(len(frames) / 6) - 7
    a = np.zeros(size)
    print(len(a))
    for i in range(40, 45):
        a[i] = 1

    print(a)

    np.save(r"C:\Users\DongHoon\Documents\PROGRAPHY DATA_ver2\testRV\testRV00(42,120)", a)
    test = TestViewer(test_video, extracted_hv)
    test.show() #show(0)show(1)show(2) 다 됨
