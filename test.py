import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import cv2
from PIL import Image
from flask import Flask
from io import BytesIO

#################################################################
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os
import time
import torch
import tensorflow as tf

######################3 Bien bao

ROOT = os.path.dirname(os.path.abspath(__file__))+'/'
model_path = ROOT + "models/mb2-ssd-lite-Epoch-325-Loss-1.11.pth"
label_path = ROOT + "models/open-images-model-labels.txt"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)

predictor = None
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print("GPU")
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=torch.device('cuda'))
else:
    print("CPU")
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=torch.device('cpu'))

#
TRAFFIC_SIGN = [0, 0, 0, 0, 0, 0, 0, 0, 0]
TRAFFIC_SIGN_TIME = [0, 0, 0, 0, 0, 0, 0, 0, 0]
TIME_DEFF = 1000
TIME_DE = 0
TIME_DE_D = 10
##############################################################


# ------------- Add library ------------#
from keras.models import load_model
import utils

# --------------------------------------#

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)

bbn = 0
tt = 0
stop = 0
IMAGE = None
color = [100, 200, 100]


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global folder_name, count, save, TRAFFIC_SIGN, TRAFFIC_SIGN_TIME, TIME_DEFF, TIME_DE, TIME_DE_D, IMAGE, bbn, tt, stop
    if data:

        steering_angle = 0  # Góc lái hiện tại của xe
        speed = 0  # Vận tốc hiện tại của xe
        image = 0  # Ảnh gốc

        steering_angle = float(data["steering_angle"])
        speed = float(data["speed"])
        # Original Image
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image0 = np.asarray(image)
        detect_traff(image0)
        orig_image = image0.copy()
        """
        - Chương trình đưa cho bạn 3 giá trị đầu vào:
            * steering_angle: góc lái hiện tại của xe
            * speed: Tốc độ hiện tại của xe
            * image: hình ảnh trả về từ xe

        - Bạn phải dựa vào 3 giá trị đầu vào này để tính toán và gửi lại góc lái và tốc độ xe cho phần mềm mô phỏng:
            * Lệnh điều khiển: send_control(sendBack_angle, sendBack_Speed)
            Trong đó:
                + sendBack_angle (góc điều khiển): [-25, 25]  NOTE: ( âm là góc trái, dương là góc phải)
                + sendBack_Speed (tốc độ điều khiển): [-150, 150] NOTE: (âm là lùi, dương là tiến)
        """
        sendBack_angle = 0
        sendBack_Speed = 0
        if True:
        # ------------------------------------------  Work space  ----------------------------------------------#
            bb = 0
            if TRAFFIC_SIGN[6] == 1 or TRAFFIC_SIGN[1] == 1 or TRAFFIC_SIGN[2] == 1 or TRAFFIC_SIGN[8] == 1 or TRAFFIC_SIGN[3] == 1 or TRAFFIC_SIGN[5] == 1:
                bb = 0

            if TRAFFIC_SIGN[0] == 1 :
                bb = 1

            if TRAFFIC_SIGN[4] == 1:
                bb = 2

            # bb = 0
            # if n==ord('1'):
            #     bb = 1
            # elif n==ord('2'):
            #     bb = 2
            # elif n == ord('s'):
            #     stop = 2
            # elif n == ord('d'):
            #     stop = 1
            # else:
            #     bb = 0

            if bbn == 0:
                bbn = bb

            x = np.array([bbn])
            # print('*****************************************************')
            image0 = cv2.resize(image0[100:,:,:], (128, 64), cv2.INTER_AREA)
            image0 = np.array([image0])
            steering_angle = float(model.predict([image0, x], batch_size=1)) * 25

            if tt == 1 and abs(steering_angle) < 5 and bbn != 0:
                tt = 0
                bbn = 0
            print ("bbn: ", bbn)
            if abs(steering_angle) > 13 and bbn != 0:
                tt = 1
            print ('tt: ', tt)
            if stop == 2:
                sendBack_Speed = (10 - speed) * 30
            elif stop == 1:
                sendBack_Speed = (0 - speed) * 100
            else:
                if bbn != 0:
                    sendBack_Speed = (10 - 0.008 * (steering_angle ** 2) - speed) * 30
                else:
                    sendBack_Speed = (30 - 0.032 * (steering_angle ** 2) - speed) * 70

        # print(bbn)
        # ------------------------------------------------------------------------------------------------------#
        # print('{} : {}'.format(sendBack_angle, sendBack_Speed))
        send_control(steering_angle, sendBack_Speed)
        # except Exception as e:

        #     print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)
 
@ sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


from threading import Thread
import time


# Define a function for the thread
def detect_traff(image):
    global folder_name, count, save, TRAFFIC_SIGN, TRAFFIC_SIGN_TIME, TIME_DEFF, TIME_DE, TIME_DE_D

    orig_image = image.copy()
    _ct = int(time.time() * 1000)
    if _ct - TIME_DE > TIME_DE_D:
        TIME_DE = _ct
        boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)
        TRAFFIC_SIGN_NEW = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(boxes.size(0)):
            label = str(f"{class_names[labels[i]]}: {probs[i]:.2f}").split(':')[0]
            print(label)
            if label == "cod_0":
                TRAFFIC_SIGN_NEW[0] = 1
            elif label == "cod_1":
                TRAFFIC_SIGN_NEW[1] = 1
            elif label == "cod_2":
                TRAFFIC_SIGN_NEW[2] = 1
            elif label == "cod_3":
                TRAFFIC_SIGN_NEW[3] = 1
            elif label == "cod_4":
                TRAFFIC_SIGN_NEW[4] = 1
            elif label == "cod_5":
                TRAFFIC_SIGN_NEW[5] = 1
            elif label == "cod_6":
                TRAFFIC_SIGN_NEW[6] = 1
            elif label == "cod_7":
                TRAFFIC_SIGN_NEW[7] = 1
            elif label == "cod_8":
                TRAFFIC_SIGN_NEW[8] = 1
            box = boxes[i, :]

            #label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

#            cv2.putText(orig_image, label,
#                         (box[0] + 20, box[1] + 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,  # font scale
#                         (255, 0, 255),
#                         2)  # line type
        for j, flag in enumerate(TRAFFIC_SIGN_NEW):
            if flag == 0 and TRAFFIC_SIGN[j] == 1:
                _ct = int(time.time() * 1000)
                if TRAFFIC_SIGN_TIME[j] == -1:
                    TRAFFIC_SIGN_TIME[j] = _ct
                elif _ct - TRAFFIC_SIGN_TIME[j] > TIME_DEFF:
                    TRAFFIC_SIGN_TIME[j] = -1
                    TRAFFIC_SIGN[j] = 0
            elif flag == 1:
                TRAFFIC_SIGN[j] = 1
                TRAFFIC_SIGN_TIME[j] = -1
    # print(TRAFFIC_SIGN)

    # cv2.imshow('Traf', orig_image)
    # n = int(cv2.waitKey(1))


if __name__ == '__main__':
    # -----------------------------------  Setup  ------------------------------------------#
    from model.Angle_model_v3 import Angle_model_v3

    model = Angle_model_v3((64, 128, 3)).build()
    model.load_weights(ROOT + 'model-107.h5')
    # --------------------------------------------------------------------------------------#
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

