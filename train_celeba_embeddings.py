import argparse
import os
import torch
from pnpflow.dataloaders import DataLoaders
from pnpflow.perceptual import LPIPSFeatureExtractor, PerceptualLoss


def _extract_images_from_batch(batch):
    if isinstance(batch, dict):
        return batch.get('y_gt', None)
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return batch[0]
    return None


def _extract_vgg_features(perceptual_criterion, imgs, v_mean, v_std):
    # Legacy VGG16 slices used by the original PerceptualLoss class.
    imgs_norm = (imgs + 1.0) / 2.0
    imgs_norm = (imgs_norm - v_mean) / v_std

    h1 = perceptual_criterion.slice1(imgs_norm)
    h2 = perceptual_criterion.slice2(h1)
    h3 = perceptual_criterion.slice3(h2)
    h4 = perceptual_criterion.slice4(h3)
    return [h1, h2, h3, h4]


def train_perceptual_embeddings(
    batch_size=32,
    max_images=1000,
    save_path='./model/celeba/mean_embeddings_celeba.pt',
    feature_backbone='lpips',
    lpips_net='vgg',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup DataLoader for CelebA
    # We use a small batch size and just a few iterations to compute the mean
    # If the user wants to "train" a feature extractor, that's different,
    # but based on the previous context, we need the "mean embeddings" 
    # of the CelebA dataset to use as a prior.
    
    print("Loading CelebA dataset...")
    loaders = DataLoaders('celeba', batch_size, batch_size)
    data_loaders = loaders.load_data()
    train_loader = data_loaders.get('train', data_loaders.get('train_loader'))
    if train_loader is None:
        raise KeyError(
            f"Could not find train loader in returned keys: {list(data_loaders.keys())}. "
            "Expected key 'train'."
        )
    
    use_lpips = feature_backbone.lower() == 'lpips'
    feature_extractor = None
    perceptual_criterion = None
    v_mean = None
    v_std = None

    # 2. Initialize feature extractor
    if use_lpips:
        try:
            feature_extractor = LPIPSFeatureExtractor(net=lpips_net, device=device)
            print(f"Using LPIPS feature extractor (net='{lpips_net}').")
        except ImportError as exc:
            print(f"Warning: {exc}")
            print("Falling back to legacy VGG16 feature slices.")
            use_lpips = False

    if not use_lpips:
        perceptual_criterion = PerceptualLoss(device=device)
        perceptual_criterion.eval()
        v_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        v_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print("Using legacy VGG16 feature slices.")

    # 3. Accumulate Embeddings
    print("Computing mean embeddings over CelebA...")
    feature_sums = None
    total_images = 0

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            imgs = _extract_images_from_batch(batch)

            if imgs is None or imgs.numel() == 0:
                continue

            imgs = imgs.to(device)

            # Respect max_images exactly by trimming the last batch if needed.
            remaining = max_images - total_images
            if remaining <= 0:
                break
            if imgs.size(0) > remaining:
                imgs = imgs[:remaining]

            if use_lpips:
                features = feature_extractor.extract_features(imgs, num_layers=4)
            else:
                features = _extract_vgg_features(perceptual_criterion, imgs, v_mean, v_std)

            if feature_sums is None:
                feature_sums = [feat.sum(dim=0, keepdim=True) for feat in features]
            else:
                for idx, feat in enumerate(features):
                    feature_sums[idx] += feat.sum(dim=0, keepdim=True)

            total_images += imgs.size(0)
            if i % 10 == 0:
                print(f"Processed {total_images} images...")
            
            if total_images >= max_images:
                break

    if total_images == 0:
        raise RuntimeError(
            "No images were processed from the train loader. "
            "Please verify the CelebA paths and partition CSV."
        )

    mean_embeddings = [feat_sum / total_images for feat_sum in feature_sums]

    # 4. Save the results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mean_embeddings, save_path)
    print(f"Mean embeddings saved to {save_path}")
    print(f"Feature backbone: {'LPIPS-' + lpips_net if use_lpips else 'VGG16-slices'}")
    print(f"Total images used: {total_images}")


def parse_args():
    parser = argparse.ArgumentParser(description='Compute CelebA mean perceptual embeddings.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-images', type=int, default=1000)
    parser.add_argument('--save-path', type=str, default='./model/celeba/mean_embeddings_celeba.pt')
    parser.add_argument('--feature-backbone', choices=['lpips', 'vgg'], default='lpips')
    parser.add_argument('--lpips-net', choices=['vgg', 'alex', 'squeeze'], default='vgg')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_perceptual_embeddings(
        batch_size=args.batch_size,
        max_images=args.max_images,
        save_path=args.save_path,
        feature_backbone=args.feature_backbone,
        lpips_net=args.lpips_net,
    )
