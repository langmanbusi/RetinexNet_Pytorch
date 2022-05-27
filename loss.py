import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    return grad_out


def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)


def smooth(input1, input2):
    input2 = 0.299 * input2[:, 0, :, :] + 0.587 * input2[:, 1, :, :] + 0.114 * input2[:, 2, :, :]
    input2 = torch.unsqueeze(input2, dim=1)
    return torch.mean(gradient(input1, "x") * torch.exp(-10 * ave_gradient(input2, "x")) +
                      gradient(input1, "y") * torch.exp(-10 * ave_gradient(input2, "y")))


def l1loss(input1, input2):
    return F.l1_loss(input1, input2)


