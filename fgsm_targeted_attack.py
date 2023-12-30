# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# FGSM targeted attack
def fgsm_targeted_attack(e=8/255, offset=49):
    atk = torchattacks.FGSM(model, eps=e)
    atk.set_mode_targeted_by_label(quiet=True)
    # 攻击目标为offset，默认为49，即African_crocodile，尼罗鳄
    target_labels = labels - labels + offset
    fgsm_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
    fgsm_targeted_pre = get_pred(model, fgsm_targeted_image[0:1], device)
    print("FGSM targeted attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[fgsm_targeted_pre.item()])
    imshow(fgsm_targeted_image[0:1], title="FGSM targeted\n" + r"$\epsilon$ = %.4f" % e + "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[fgsm_targeted_pre.item()]))

# fgsm_targeted_attack() # fgsm-targeted-attacked-default.png
fgsm_targeted_attack(e=0.0001) # fgsm-targeted-attacked-0.0001.png
fgsm_targeted_attack(e=0.001) # fgsm-targeted-attacked-0.001.png
fgsm_targeted_attack(e=0.01) # fgsm-targeted-attacked-0.01.png
fgsm_targeted_attack(e=0.1) # fgsm-targeted-attacked-0.1.png
fgsm_targeted_attack(e=0.5) # fgsm-targeted-attacked-0.5.png
fgsm_targeted_attack(e=1) # fgsm-targeted-attacked-1.png