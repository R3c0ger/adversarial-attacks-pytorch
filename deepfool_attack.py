# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# DeepFool attack
def deepfool_attack(s=50, o=0.20):
    atk = torchattacks.DeepFool(model, steps=s, overshoot=o)
    deepfool_image = atk(images[idx:idx+1], labels[idx:idx+1])
    deepfool_pre = get_pred(model, deepfool_image[0:1], device)
    print("DeepFool attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[deepfool_pre.item()])
    imshow(deepfool_image[0:1], title="DeepFool\nsteps = %d, overshoot = %.2f" % (s, o)+ "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[deepfool_pre.item()]))

deepfool_attack(s=10, o=0.20) # deepfool-attacked-default.png
deepfool_attack(s=1, o=0.20) # deepfool-attacked-1.png
deepfool_attack(s=11, o=0.20) # deepfool-attacked-11.png
deepfool_attack(s=21, o=0.20) # deepfool-attacked-21.png
deepfool_attack(s=31, o=0.20) # deepfool-attacked-31.png
deepfool_attack(s=41, o=0.20) # deepfool-attacked-41.png
deepfool_attack(s=50, o=0.20) # deepfool-attacked-50.png