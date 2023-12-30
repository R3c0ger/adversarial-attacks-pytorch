# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# PGD targeted attack
def pgd_targeted_attack(e=8/255, offset=49, a=2/225, s=10, rand_start=True):
    atk = torchattacks.PGD(model, eps=e, alpha=a, steps=s, random_start=rand_start)
    atk.set_mode_targeted_by_label(quiet=True)
    target_labels = labels - labels + offset
    pgd_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
    pgd_targeted_pre = get_pred(model, pgd_targeted_image[0:1], device)
    print("PGD targeted attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[pgd_targeted_pre.item()])
    imshow(pgd_targeted_image[0:1], title="PGD targeted\n" + r"$\epsilon$ = %.4f" % e + "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[pgd_targeted_pre.item()]))

pgd_targeted_attack() # pgd-targeted-attacked-default.png
pgd_targeted_attack(e=0.0001) # pgd-targeted-attacked-0.0001.png
pgd_targeted_attack(e=0.1) # pgd-targeted-attacked-0.1.png
pgd_targeted_attack(e=0.2) # pgd-targeted-attacked-0.2.png
pgd_targeted_attack(e=0.3) # pgd-targeted-attacked-0.3.png
pgd_targeted_attack(e=0.4) # pgd-targeted-attacked-0.4.png
pgd_targeted_attack(e=0.5) # pgd-targeted-attacked-0.5.png