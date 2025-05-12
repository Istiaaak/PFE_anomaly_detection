import torch
from torch import tensor
from torchvision import transforms

import numpy as np
import PIL
from PIL import ImageFilter
from sklearn import random_projection
from tqdm import tqdm


from data.data import mvtec_classes


backbones = {
    'WideResNet50':'wide_resnet50_2',
    'ResNet101':'RN101',
    'ResNet50':'RN50',
    'ResNet50-4':'RN50x4',
    'ResNet50-16':'RN50x16',
}

dataset_scale_factor = {
    'WideResNet50': 1,
    'ResNet101': 1,
    'ResNet50': 1,
    'ResNet50-4': 2,
    'ResNet50-16': 4,
}

def get_coreset(
        memory_bank: tensor,
        l: int = 1000,
        eps: float = 0.09,
) -> tensor:

    coreset_idx = []
    idx = 0

    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        memory_bank = torch.tensor(transformer.fit_transform(memory_bank))
    except ValueError:
        print("Error: could not project vectors. Please increase `eps`.")

    print(f'Start Coreset Subsampling...')

    last_item = memory_bank[idx: idx + 1]   
    coreset_idx.append(torch.tensor(idx))
    min_distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True) 

    if torch.cuda.is_available():
        last_item = last_item.to("cuda")
        memory_bank = memory_bank.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(l - 1)):
        distances = torch.linalg.norm(memory_bank - last_item, dim=1, keepdims=True)  
        min_distances = torch.minimum(distances, min_distances)                         
        idx = torch.argmax(min_distances)                                             

        last_item = memory_bank[idx: idx + 1]  
        min_distances[idx] = 0                
        coreset_idx.append(idx.to("cpu"))       

    return torch.stack(coreset_idx)


def gaussian_blur(img: tensor) -> tensor:

    blur_kernel = ImageFilter.GaussianBlur(radius=4)
    tensor_to_pil = transforms.ToPILImage()
    pil_to_tensor = transforms.ToTensor()


    max_value = img.max()  
    blurred_pil = tensor_to_pil(img[0] / max_value).filter(blur_kernel)
    blurred_map = pil_to_tensor(blurred_pil) * max_value

    return blurred_map


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
        
    return PIL.Image.fromarray(tensor)


def display_backbones():
    vanilla = True
    print("Vanilla PatchCore backbone:")
    print(f"- WideResNet50")
    print("CLIP Image Encoder architectures for PatchCore backbone:")
    for k, _ in backbones.items():
        if vanilla:
            vanilla = False
            continue
        print(f"- {k}")
    print()
    

def display_MVTec_classes():
    print(mvtec_classes())

