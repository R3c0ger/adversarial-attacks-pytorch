# -*- coding: utf-8 -*-
from imagenet_test import model, images, labels, idx2label, device, idx, fgsm_attack
import torchattacks
from demo.utils import imshow, get_pred
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# 分析改变扰动 𝜖 的大小对对抗图像可视性和攻击成功率。
def fgsm_attack_analysis(start=0.00, end=1.00, step=0.01):
    eps_num = int((end - start) / step + 2) # 计算 𝜖 取值个数 + 1
    correct = [0 for i in range(eps_num)] # 预测正确的图片数
    total = [0 for i in range(eps_num)] # 总共的图片数
    eps_list = [0 for i in range(eps_num)] # 扰动的大小
    # 扰动 𝜖 的取值范围为 [start, end]，步长为 step，递增变化，计算每个 𝜖 对应的攻击成功率
    for i, e in enumerate(np.arange(start, end + 1, step)):
        e = float(Decimal(str(e)).quantize(Decimal('0.0000')))
        eps_list[i] = e if e <= end else end
        atk = torchattacks.FGSM(model, eps=eps_list[i])
        fgsm_image = atk(images, labels)
        # 对 25 张图片进行攻击，计算攻击成功率
        for idx in range(len(images)):
            fgsm_pre = get_pred(model, fgsm_image[idx:idx+1], device)
            # print("FGSM attack: original label", idx2label[labels[idx:idx + 1].item()], " pred label", idx2label[fgsm_pre.item()])
            if idx2label[labels[idx:idx+1].item()] == idx2label[fgsm_pre.item()]:
                correct[i] += 1
            total[i] += 1
        print("eps:", eps_list[i], "correct:", correct[i], "total:", total[i], "accuracy:", correct[i]/total[i])
        if eps_list[i] == end:
            break
    # 将攻击成功率绘制成图像
    plt.plot(eps_list, np.array(correct) / np.array(total), marker= '^')
    plt.grid(ls='--')
    plt.title("FGSM Attack")
    plt.xlabel(r'$\epsilon$')
    plt.ylabel("Accuracy")
    plt.savefig("../img/fgsm-analysis.png")
    plt.show()

fgsm_attack_analysis(start=0, end=0.003, step=0.0003)