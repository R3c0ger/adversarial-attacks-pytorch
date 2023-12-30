# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# FGSM attack
def fgsm_attack(e=8/255):
    # 避免报错：RuntimeError: expected scalar type Double but found Float，添加参数e
    atk = torchattacks.FGSM(model, eps=e) # atk 表示攻击方法
    # fgsm_image 是攻击后生成的图片集。由于DeepFool等的攻击的时间较长，所以只取一张图片(idx)进行攻击
    fgsm_image = atk(images[idx:idx+1], labels[idx:idx+1])
    fgsm_pre = get_pred(model, fgsm_image[0:1], device) # 获取攻击后图片预测的标签。只攻击了部分图片，故图片index不为[idx:idx+1]
    print("FGSM attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[fgsm_pre.item()])
    imshow(fgsm_image[0:1], title="FGSM\n" + r"$\epsilon$ = %.4f" % e + "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[fgsm_pre.item()]))  # 攻击后的图片

fgsm_attack(e=8/255) # fgsm-attacked-default

fgsm_attack(e=0.0003) # fgsm_0.0003.png
fgsm_attack(e=0.0030) # fgsm_0.003.png
fgsm_attack(e=0.0300) # fgsm_0.03.png
fgsm_attack(e=0.3000) # fgsm_0.3.png
fgsm_attack(e=1.0000) # fgsm_1.png
