# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx, pgd_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def pgd_attack_analysis(start=0.00, end=1.00, step=0.01):
    eps_num = int((end - start) / step + 2)
    correct = [0 for i in range(eps_num)]
    total = [0 for i in range(eps_num)]
    eps_list = [0 for i in range(eps_num)]
    for i, e in enumerate(np.arange(start, end + 1, step)):
        e = float(Decimal(str(e)).quantize(Decimal('0.0000')))
        eps_list[i] = e if e <= end else end
        atk = torchattacks.PGD(model, eps=eps_list[i], alpha=2/225, steps=10, random_start=True)
        pgd_image = atk(images, labels)
        for idx in range(len(images)):
            pgd_pre = get_pred(model, pgd_image[idx:idx+1], device)
            # print("PGD attack: original label", idx2label[labels[idx:idx + 1].item()], " pred label", idx2label[pgd_pre.item()])
            # imshow(pgd_image[idx:idx+1], title="PGD True:%s, Pre:%s"%(idx2label[labels[idx:idx+1].item()], idx2label[pgd_pre.item()]))
            # if idx == 0:
            #     imshow(pgd_image[idx:idx+1], title="PGD True:%s, Pre:%s"%(idx2label[labels[idx:idx+1].item()], idx2label[pgd_pre.item()]))
            if idx2label[labels[idx:idx+1].item()] == idx2label[pgd_pre.item()]:
                correct[i] += 1
            total[i] += 1
        print("eps:", eps_list[i], "correct:", correct[i], "total:", total[i], "accuracy:", correct[i]/total[i])
        if eps_list[i] == end:
            break
    plt.plot(eps_list, np.array(correct) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("PGD Attack")
    plt.xlabel(r'$\epsilon$')
    plt.ylabel("Accuracy")
    plt.savefig("../img/pgd-analysis.png")
    plt.show()

pgd_attack_analysis(start=0, end=0.0028, step=0.0003)