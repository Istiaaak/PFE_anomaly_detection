import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.data import MVTecDataset, DEFAULT_SIZE
from model.patch_core import PatchCore
from utils.utils import backbones, dataset_scale_factor
from tqdm import tqdm
from typing import Optional, Callable

def tensor_to_img(x: torch.Tensor, vanilla: bool) -> np.ndarray:
    x = x.clone().cpu()
    if vanilla:
        mean = torch.tensor([.485, .456, .406])
        std  = torch.tensor([.229, .224, .225])
    else:
        mean = torch.tensor([.481, .457, .408])
        std  = torch.tensor([.268, .261, .275])
    for c in range(x.shape[0]):
        x[c] = x[c] * std[c] + mean[c]
    return x.clamp(0.0, 1.0).permute(1, 2, 0).numpy()

def load_patchcore_model(
    cls: str,
    backbone_key: str,
    f_coreset: float,
    eps: float,
    k_nn: int,
    use_cache: bool,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    size    = DEFAULT_SIZE
    vanilla = (backbone_key == 'WideResNet50')
    ds      = MVTecDataset(cls, size=size, vanilla=vanilla)
    train_ds, _ = ds.get_datasets()
    train_dl = DataLoader(train_ds, batch_size=1)

    model = PatchCore(
        f_coreset   = f_coreset,
        eps_coreset = eps,
        k_nearest   = k_nn,
        vanilla     = vanilla,
        backbone    = backbones[backbone_key],
        image_size  = size
    )

    cache_file = Path("./patchcore_cache/memory_bank") / f"{cls}_{backbone_key}_f{f_coreset:.3f}.pth"
    if use_cache and cache_file.exists():
        print(f"[load_patchcore_model] Chargement de la memory_bank depuis {cache_file}", flush=True)
        mb = torch.load(cache_file)
        model.memory_bank = mb if isinstance(mb, torch.Tensor) else torch.cat(mb, 0)
        model.avg = torch.nn.AvgPool2d(3, stride=1)
        batch, _ = next(iter(train_dl))
        _ = model.forward(batch)
        fmap_size = model.features[0].shape[-2]
        model.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
    else:
        print("[load_patchcore_model] Entraînement de la memory_bank...", flush=True)
        model.fit(train_dl, scale=dataset_scale_factor[backbone_key])
        mb = model.memory_bank.cpu() if isinstance(model.memory_bank, torch.Tensor) else torch.cat(model.memory_bank,0).cpu()
        torch.save(mb, cache_file)
        print(f"[load_patchcore_model] Memory_bank sauvée dans {cache_file}", flush=True)

    print("[load_patchcore_model] Calibration du seuil sur les images good …", flush=True)
    train_scores = []
    total = len(train_dl)
    for i, (x, _) in enumerate(tqdm(train_dl, total=total, desc="Calibrating threshold", unit="img")):
        s, _ = model.predict(x)
        train_scores.append(s.item())
        if progress_callback:
            progress_callback(i + 1, total)

    train_scores = np.array(train_scores)
    default_thresh = float(np.percentile(train_scores, 90))
    print(f"[load_patchcore_model] Seuil par défaut (90e percentile) = {default_thresh:.4f}", flush=True)

    return model, train_scores