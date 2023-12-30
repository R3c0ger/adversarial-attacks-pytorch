# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx, deepfool_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import time

def deepfool_attack_analysis(start=0, end=50, step=1):
    steps_num = int((end - start) / step + 1)
    correct = [0 for i in range(steps_num)]
    total = [0 for i in range(steps_num)]
    steps_list = [0 for i in range(steps_num)]
    for i, s in enumerate(np.arange(start, end + 1, step)):
        start_time = time.time()
        steps_list[i] = s if s <= end else end
        atk = torchattacks.DeepFool(model, steps=s, overshoot=0.20)
        deepfool_image = atk(images[0:16], labels[0:16])
        for idx in range(len(images[0:16])):
            deepfool_pre = get_pred(model, deepfool_image[idx:idx+1], device)
            print("DeepFool attack: original label", idx2label[labels[idx:idx + 1].item()], " pred label", idx2label[deepfool_pre.item()])
            # imshow(deepfool_image[idx:idx+1], title="DeepFool True:%s, Pre:%s"%(idx2label[labels[idx:idx+1].item()], idx2label[deepfool_pre.item()]))
            # if idx == 0:
            #     imshow(deepfool_image[idx:idx+1], title="DeepFool True:%s, Pre:%s"%(idx2label[labels[idx:idx+1].item()], idx2label[deepfool_pre.item()]))
            if idx2label[labels[idx:idx+1].item()] == idx2label[deepfool_pre.item()]:
                correct[i] += 1
            total[i] += 1
        end_time = time.time()
        print("steps:", steps_list[i], "correct:", correct[i], "total:", total[i], "accuracy:", correct[i]/total[i])
        print("time:", Decimal(str(end_time - start_time)).quantize(Decimal('0.00000000')))
        if steps_list[i] == end:
            break
    plt.plot(steps_list, np.array(correct) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("DeepFool Attack")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.savefig("../img/deepfool-analysis.png")
    plt.show()

deepfool_attack_analysis(start=0, end=5, step=1)