import torch
from torch import tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchvision
import torchvision.transforms as T

from tqdm import tqdm
import clip 
from PIL import Image 
import numpy as np
from sklearn.metrics import roc_auc_score

from utils.utils import gaussian_blur, get_coreset


class PatchCore(torch.nn.Module):
    def __init__(
            self,
            f_coreset:float = 0.01,    # Fraction rate of training samples
            eps_coreset: float = 0.90, # SparseProjector parameter
            k_nearest: int = 3,        # k parameter for K-NN search
            vanilla: bool = True,      # if False, use CLIP
            backbone: str = 'wide_resnet50_2',
            image_size: int = 224
    ):
        assert f_coreset > 0
        assert eps_coreset > 0
        assert k_nearest > 0
        assert image_size > 0

        super(PatchCore, self).__init__()

        if vanilla:
            print(f"Vanilla Mode")
        else: 
            print(f"CLIP Mode")
        print(f"Net Used: {backbone}")

        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.featured."""
            self.features.append(output)

        if vanilla == True:
            self.model = torch.hub.load('pytorch/vision:v0.13.0', 'wide_resnet50_2', pretrained=True)
            self.model.layer2[-1].register_forward_hook(hook)            
            self.model.layer3[-1].register_forward_hook(hook)            
        else:
            self.model, _ = clip.load(backbone, device="cpu")
            self.model.visual.layer2[-1].register_forward_hook(hook)
            self.model.visual.layer3[-1].register_forward_hook(hook)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.memory_bank = []
        self.f_coreset = f_coreset
        self.eps_coreset = eps_coreset
        self.k_nearest = k_nearest     
        self.vanilla = vanilla          
        self.backbone = backbone
        self.image_size = image_size


    def forward(self, sample: tensor):
        self.features = []
        if self.vanilla:
            _ = self.model(sample)
        else:
            _ = self.model.visual(sample)

        return self.features


    def fit(self, train_dataloader: DataLoader, scale: int=1) -> None:
        tot = len(train_dataloader) // scale
        counter = 0

        for sample, _ in tqdm(train_dataloader, total=tot):
            feature_maps = self(sample)                 

            self.avg = torch.nn.AvgPool2d(3, stride=1)
            fmap_size = feature_maps[0].shape[-2]        
            self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

            # Create patch
            resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
            patch = torch.cat(resized_maps, 1)           
            patch = patch.reshape(patch.shape[1], -1).T  

            self.memory_bank.append(patch)       
            counter += 1
            if counter > tot:
                break

        self.memory_bank = torch.cat(self.memory_bank, 0)

        # Coreset subsampling
        if self.f_coreset < 1:
            coreset_idx = get_coreset(
                self.memory_bank,
                l = int(self.f_coreset * self.memory_bank.shape[0]),
                eps = self.eps_coreset
            )
            self.memory_bank = self.memory_bank[coreset_idx]


    def evaluate(self, test_dataloader: DataLoader):

        image_preds = []
        image_labels = []
        pixel_preds = []
        pixel_labels = []

        for sample, mask, label in tqdm(test_dataloader):

            image_labels.append(label)
            pixel_labels.extend(mask.flatten().numpy())

            score, segm_map = self.predict(sample)

            image_preds.append(score.numpy())
            pixel_preds.extend(segm_map.flatten().numpy())

        image_labels = np.stack(image_labels)
        image_preds = np.stack(image_preds)

        image_level_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_level_rocauc = roc_auc_score(pixel_labels, pixel_preds)

        return image_level_rocauc, pixel_level_rocauc


    def predict(self, sample: tensor):
        feature_maps = self(sample)
        resized_maps = [self.resize(self.avg(fmap)) for fmap in feature_maps]
        patch = torch.cat(resized_maps, 1)
        patch = patch.reshape(patch.shape[1], -1).T

        distances = torch.cdist(patch, self.memory_bank, p=2.0)
        dist_score, dist_score_idxs = torch.min(distances, dim=1)
        s_idx = torch.argmax(dist_score)
        s_star = torch.max(dist_score)
        m_test_star = torch.unsqueeze(patch[s_idx], dim=0)
        m_star = self.memory_bank[dist_score_idxs[s_idx]].unsqueeze(0)

        knn_dists = torch.cdist(m_star, self.memory_bank, p=2.0)
        _, nn_idxs = knn_dists.topk(k=self.k_nearest, largest=False)

        m_star_neighbourhood = self.memory_bank[nn_idxs[0, 1:]]
        w_denominator = torch.linalg.norm(m_test_star - m_star_neighbourhood, dim=1)
        norm = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / norm) / torch.sum(torch.exp(w_denominator / norm)))
        s = w * s_star

        fmap_size = feature_maps[0].shape[-2:]
        segm_map = dist_score.view(1, 1, *fmap_size)
        segm_map = torch.nn.functional.interpolate(
                        segm_map,
                        size=(self.image_size, self.image_size),
                        mode='bilinear'
                    )
        segm_map = gaussian_blur(segm_map)            

        return s, segm_map