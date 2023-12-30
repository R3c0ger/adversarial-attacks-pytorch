# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx
import torchattacks
from demo.utils import imshow, get_pred

# JSMA targeted attack
def jsma_targeted_attack(t=1.0, g=0.1, offset=49):
    atk = torchattacks.JSMA(model, theta=t, gamma=g)
    atk.set_mode_targeted_by_label(quiet=True)
    target_labels = labels - labels + offset
    jsma_targeted_image = atk(images[idx:idx + 1], target_labels[idx:idx + 1])
    jsma_targeted_pre = get_pred(model, jsma_targeted_image[0:1], device)
    print("JSMA targeted attack: original label", idx2label[labels[idx:idx+1].item()], " pred label", idx2label[jsma_targeted_pre.item()])
    imshow(jsma_targeted_image[0:1], title="JSMA targeted\n" + r"$\theta$ = %.4f, $\gamma$ = %.4f" % (t, g) + "\nTrue:%s, Pre:%s" % (
        idx2label[labels[idx:idx + 1].item()], idx2label[jsma_targeted_pre.item()]))

jsma_targeted_attack() # jsma-targeted-attacked-default
