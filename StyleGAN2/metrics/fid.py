import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from StyleGAN2.Model import Generator
from .iv3_utility import load_inception_v3


@torch.no_grad()
def extract_features_from_samples(
        generator: Generator, inception, batch_size, sample_no, truncation, trunc_latent, device):
    batch_no = sample_no // batch_size
    left = sample_no - (batch_no * batch_size)
    batch_sizes = [batch_size] * batch_no + [left]
    ex_features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = generator([latent], truncation=truncation, truncate_latent=trunc_latent)
        feature = inception(img)[0].view(img.shape[0], -1)

        ex_features.append(feature.to("cpu"))

    ex_features = torch.cat(ex_features, dim=0)
    return ex_features


def frechet_inception_distance(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("Product of covariation matrices has singularities")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"There is a imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
    fid = mean_norm + trace
    return fid
