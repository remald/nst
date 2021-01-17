import copy

import torch
from torch import optim, nn
import torchvision.models as models

from loss import ContentLoss, StyleLoss, TotalVariationLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(pretrained=True).features.to(device).eval()


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=0.05)
    return optimizer


def get_style_model_and_losses(cnn, style_img1, style_img2, content_img,
                               content_layers=None,
                               style_layers=None):
    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    if content_layers is None:
        content_layers = ['conv_4']

    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization().to(device)

    # just in order to have an iterable access to or list of content/syle/tv
    # losses
    content_losses = []
    style_losses = []
    tv_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature1 = model(style_img1).detach()
            target_feature2 = model(style_img2).detach()
            style_loss = StyleLoss(target_feature1, target_feature2)
            tv_loss = TotalVariationLoss()
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            # to every style loss layer add TV regularization loss
            model.add_module("tv_loss_{}".format(i), tv_loss)
            tv_losses.append(tv_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses, tv_losses
