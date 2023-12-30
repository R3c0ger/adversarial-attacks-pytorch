# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx, fgsm_targeted_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def fgsm_targeted_attack_accuracy(start=0.00, end=0.06, step=0.004, offset=49):
    eps_num = int((end - start) / step + 2)
    eps_list = [0 for i in range(eps_num)]
    total = [0 for i in range(eps_num)]
    accurate = [0 for i in range(eps_num)]
    for i, e in enumerate(np.arange(start, end + 1, step)):
        e = float(Decimal(str(e)).quantize(Decimal('0.0000')))
        eps_list[i] = e if e <= end else end
        atk = torchattacks.FGSM(model, eps=eps_list[i])
        atk.set_mode_targeted_by_label(quiet=True)
        target_labels = labels - labels + offset
        for idx in range(len(images)):
            fgsm_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
            fgsm_targeted_pre = get_pred(model, fgsm_targeted_image[0:1], device)
            # print("FGSM targeted attack: original label", idx2label[labels[idx:idx+1].item()],
            #       " target", idx2label[target_labels[idx:idx + 1].item()],
            #       " pred label", idx2label[fgsm_targeted_pre.item()])
            if idx2label[target_labels[idx:idx + 1].item()] == idx2label[fgsm_targeted_pre.item()]:
                accurate[i] += 1
            total[i] += 1
        print("eps:", eps_list[i], "correct:", accurate[i], "total:", total[i], "accuracy:", accurate[i]/24)
        if eps_list[i] == end:
            break
    plt.plot(eps_list, np.array(accurate) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("FGSM Targeted Attack")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Accuracy")
    plt.savefig("../img/fgsm-targeted-attack-accuracy.png")
    plt.show()

fgsm_targeted_attack_accuracy() # fgsm-targeted-attack-accuracy.png