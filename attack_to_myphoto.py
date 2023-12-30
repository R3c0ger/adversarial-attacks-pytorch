# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# FGSM targeted attack to my photograph
def fgsm_targeted_attack_to_myphoto(e=8/255, offset=49):
    myphoto_pre = get_pred(model, images[0:1], device) # 识别自己的照片，n01582220/n01582220_0001.JPEG
    atk = torchattacks.FGSM(model, eps=e)
    # 设定目标标签为"African_crocodile"
    atk.set_mode_targeted_by_label(quiet=True)
    target_labels = labels - labels + offset
    fgsm_targeted_image = atk(images[0:1], target_labels[0:1])
    fgsm_targeted_pre = get_pred(model, fgsm_targeted_image[0:1], device)
    print("FGSM targeted attack: original label", idx2label[myphoto_pre.item()], " pred label", idx2label[fgsm_targeted_pre.item()])
    imshow(images[0:1], title="Original\nTrue: %s"%(idx2label[myphoto_pre.item()]))
    imshow(fgsm_targeted_image[0:1], title="FGSM targeted\nTrue: %s, Pre: %s"%(idx2label[myphoto_pre.item()], idx2label[fgsm_targeted_pre.item()]))

fgsm_targeted_attack_to_myphoto(e=0.012) # 由先前的分析知，FGSM攻击的epsilon取0.012时，攻击效果最好
# myphoto-original  myphoto-fgsm-targeted