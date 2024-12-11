import torch
import torchvision.transforms as transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image


def quality_fid_scorer(images1: list, images2: list) -> float:
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    images1_tensor = []
    images2_tensor = []
    for image in images1:
        images1_tensor.append(transform(image))
    for image in images2:
        images2_tensor.append(transform(image))
    images1_tensor = torch.stack(images1_tensor)
    images2_tensor = torch.stack(images2_tensor)
    fid = FrechetInceptionDistance()
    fid.update(images1_tensor, real=True)
    fid.update(images2_tensor, real=False)
    return float(fid.compute().detach().round())


def quality_is_scorer(images: list) -> tuple:
    transform = transforms.Compose([
        transforms.PILToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images_tensor = []
    for image in images:
        images_tensor.append(transform(image.resize((299, 299), Image.BILINEAR)))
    images_tensor = torch.stack(images_tensor)
    print(images_tensor.shape)
    metric = InceptionScore()
    metric.update(images_tensor)
    score, std = metric.compute()
    return float(score.detach()), float(std.detach().round())


if __name__ == '__main__':
    img0_0 = Image.open('../373988/output_image_-1_0.png')
    img0_1 = Image.open('../373988/output_image_-1_1.png')
    img0_2 = Image.open('146656/output_image_-1_3.png')
    img0_3 = Image.open('146656/output_image_-1_4.png')
    imgs1 = [img0_0, img0_1, img0_2, img0_3]  # real image list

    img2_0 = Image.open('../373988/output_image_1_0.png')
    img2_1 = Image.open('../373988/output_image_1_1.png')
    img2_2 = Image.open('146656/output_image_1_0.png')
    img2_3 = Image.open('146656/output_image_1_1.png')
    imgs2 = [img2_0, img2_1, img2_2, img2_3]  # generated image list

    print("FID Score =", quality_fid_scorer(imgs1, imgs2))
    print("Inception Score =", quality_is_scorer(imgs2 * 3))  # require > 10 images, temporarily * 3
