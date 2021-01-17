import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Images:

    def __init__(self, content, style1, style2, size=128):
        self.transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor()])

        self.style1 = self.image_loader(style1)
        self.style2 = self.image_loader(style2)
        self.content = self.image_loader(content)

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.transforms(image).unsqueeze(0)
        return image.to(device, torch.float)


unloader = transforms.ToPILImage()
