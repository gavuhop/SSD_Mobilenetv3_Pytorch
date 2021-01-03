import cv2
import sys
from vision.utils.misc import Timer
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite,create_mobilenetv3_ssd_lite_predictor
from vision.nn.mobilenet_v3 import  MobileNetV3
import torch
if len(sys.argv) < 4:
    print('Usage: python ')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
testnet = MobileNetV3().features
# print ("test net",testnet)
net = create_mobilenetv3_ssd_lite(len(class_names), is_test=True)
net.load(model_path)

# print("total parameter :",get_model_parameters(testnet))
print("===========================================================")

predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=10)
from pytorch_model_summary import summary
# # python run_ssd_live_demo.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-25-Loss-1.8594062768045019.pth models/open-images-model-labels.txt
# print(summary(predictor.net.base_net, torch.zeros((224, 3, 3, 3)), show_input=True))
print("===========================================================")
print(predictor.net)
print("===========================================================")
print()