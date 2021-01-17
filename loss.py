import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature1, target_feature2):
        super(StyleLoss, self).__init__()
        self.target1 = self.gram_matrix(target_feature1).detach()
        self.mask1 = self.createMask_(target_feature1)
        self.target2 = self.gram_matrix(target_feature2).detach()
        self.mask2 = 1 - self.mask1
        self.loss = F.mse_loss(self.target1, self.target1)  # to initialize with something

    def forward(self, input):
        G1 = self.gram_matrix(input * self.mask1)
        G2 = self.gram_matrix(input * self.mask2)
        self.loss = F.mse_loss(G1, self.target1) + F.mse_loss(G2, self.target2)
        return input

    def createMask_(self, t):
        result = np.ones_like(t.numpy())
        result = np.triu(result)
        return torch.from_numpy(result)

    def gram_matrix(self, input):
        batch_size, h, w, f_map_num = input.size()
        features = input.view(batch_size * h, w * f_map_num)
        G = torch.mm(features, features.t())
        return G.div(batch_size * h * w * f_map_num)


class TotalVariationLoss(nn.Module):
    def __init__(self):
      super(TotalVariationLoss, self).__init__()
      self.loss = 1

    def forward(self, input):
      self.loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
      return input
