# -*- coding: utf-8 -*-
import pprint

from imagenet_test import model, images, labels, idx2label, device, idx, pgd_targeted_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# PGD targeted attack accuracy
def pgd_targeted_attack_accuracy(start=0.00, end=0.06, step=0.004, offset=49, a=2/225, s=10, rand_start=True):
    eps_num = int((end - start) / step + 1)
    eps_list = [0 for _ in range(0, eps_num)]
    total = [len(images) for _ in range(0, eps_num)]
    accurate = [0 for _ in range(0, eps_num)]
    for i, e in enumerate(np.arange(start, end + 1, step)):
        e = float(Decimal(str(e)).quantize(Decimal('0.0000')))
        eps_list[i] = e if e <= end else end
        atk = torchattacks.PGD(model, eps=eps_list[i], alpha=a, steps=s, random_start=rand_start)
        atk.set_mode_targeted_by_label(quiet=True)
        target_labels = labels - labels + offset
        for idx in range(len(images)):
            pgd_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
            pgd_targeted_pre = get_pred(model, pgd_targeted_image[0:1], device)
            # print("PGD targeted attack: original label", idx2label[labels[idx:idx+1].item()],
            #         " target", idx2label[target_labels[idx:idx + 1].item()],
            #         " pred label", idx2label[pgd_targeted_pre.item()])
            if idx2label[target_labels[idx:idx + 1].item()] == idx2label[pgd_targeted_pre.item()]:
                accurate[i] += 1
        print("eps:", eps_list[i], "correct:", accurate[i], "total:", total[i], "accuracy:", accurate[i]/24)
        if eps_list[i] == end:
            break
    plt.plot(eps_list, np.array(accurate) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("PGD Targeted Attack")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Accuracy")
    plt.savefig("../img/pgd-targeted-attack-accuracy.png")
    plt.show()

pgd_targeted_attack_accuracy(start=0.00, end=0.03, step=0.002) # pgd-targeted-attack-accuracy