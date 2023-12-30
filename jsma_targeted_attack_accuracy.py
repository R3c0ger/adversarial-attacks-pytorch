# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx, jsma_targeted_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def jsma_targeted_attack_accuracy(start=0.00, end=0.06, step=0.004, offset=49):
    theta_num = int((end - start) / step + 2)
    theta_list = [0 for i in range(theta_num)]
    total = [0 for i in range(theta_num)]
    accurate = [0 for i in range(theta_num)]
    for i, t in enumerate(np.arange(start, end + 1, step)):
        t = float(Decimal(str(t)).quantize(Decimal('0.0000')))
        theta_list[i] = t if t <= end else end
        atk = torchattacks.JSMA(model, theta=theta_list[i])
        atk.set_mode_targeted_by_label(quiet=True)
        target_labels = labels - labels + offset
        for idx in range(len(images)):
            jsma_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
            jsma_targeted_pre = get_pred(model, jsma_targeted_image[0:1], device)
            # print("JSMA targeted attack: original label", idx2label[labels[idx:idx+1].item()],
            #         " target", idx2label[target_labels[idx:idx + 1].item()],
            #         " pred label", idx2label[jsma_targeted_pre.item()])
            if idx2label[target_labels[idx:idx + 1].item()] == idx2label[jsma_targeted_pre.item()]:
                accurate[i] += 1
            total[i] += 1
        print("theta:", theta_list[i], "correct:", accurate[i], "total:", total[i], "accuracy:", accurate[i]/24)
        if theta_list[i] == end:
            break
    plt.plot(theta_list, np.array(accurate) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("JSMA Targeted Attack")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Accuracy")
    plt.savefig("../img/jsma-targeted-attack-accuracy.png")
    plt.show()

jsma_targeted_attack_accuracy() # jsma-targeted-attack-accuracy