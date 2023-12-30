# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# PGD attack
def pgd_attack(e=8/255, a=2/225, s=10, rand_start=True):
    atk = torchattacks.PGD(model, eps=e, alpha=a, steps=s, random_start=rand_start)
    pgd_image = atk(images[idx:idx+1], labels[idx:idx+1])
    pgd_pre = get_pred(model, pgd_image[0:1], device)
    print("PGD attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[pgd_pre.item()])
    imshow(pgd_image[0:1], title="PGD\n" + r"$\epsilon$ = %.4f" % e + "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[pgd_pre.item()]))

pgd_attack(e=8/255) # pgd-attacked-default
pgd_attack(e=0.0001) # pgd-attacked-0.0001.png
pgd_attack(e=0.0010) # pgd-attacked-0.001.png
pgd_attack(e=0.0100) # pgd-attacked-0.01.png
pgd_attack(e=0.1000) # pgd-attacked-0.1.png
pgd_attack(e=1.0000) # pgd-attacked-1.png
