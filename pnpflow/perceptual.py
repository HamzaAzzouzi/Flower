import torch
import torch.nn as nn
import torchvision.models as models

try:
    import lpips
except ImportError:
    lpips = None

class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss to help preserve facial features and details.
    """
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        # Load a pre-trained VGG16 model
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # We use blocks to get access to intermediate features
        # Block 1: relu1_2, Block 2: relu2_2, Block 3: relu3_3, Block 4: relu4_3
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
            
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.device = device
        self.to(device)
        self.eval()

    def forward(self, x, y):
        """
        Computes the L1 distance between VGG features of x and y.
        Expects images in range [-1, 1].
        """
        # Normalize to ImageNet statistics for VGG
        # Map [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std
        
        h_x1 = self.slice1(x)
        h_y1 = self.slice1(y)
        loss = torch.mean(torch.abs(h_x1 - h_y1))
        
        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)
        loss += torch.mean(torch.abs(h_x2 - h_y2))
        
        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)
        loss += torch.mean(torch.abs(h_x3 - h_y3))
        
        h_x4 = self.slice4(h_x3)
        h_y4 = self.slice4(h_y3)
        loss += torch.mean(torch.abs(h_x4 - h_y4))
        
        return loss


class LPIPSPerceptualLoss(nn.Module):
    """
    LPIPS perceptual distance using a pretrained backbone.
    Expects images in range [-1, 1].
    """

    def __init__(self, net='vgg', device='cuda', spatial=False):
        super().__init__()
        if lpips is None:
            raise ImportError(
                "lpips is not installed. Install it with: pip install lpips"
            )

        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, spatial=spatial).to(device)
        self.loss_fn.eval()

        # Keep LPIPS frozen as a pretrained perceptual metric.
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        return self.loss_fn(x, y).mean()


class LPIPSFeatureExtractor(nn.Module):
    """
    Extract intermediate pretrained LPIPS backbone features.
    Useful for computing dataset mean embeddings in LPIPS feature space.
    Expects images in range [-1, 1].
    """

    def __init__(self, net='vgg', device='cuda'):
        super().__init__()
        if lpips is None:
            raise ImportError(
                "lpips is not installed. Install it with: pip install lpips"
            )

        self.device = device
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def extract_features(self, x, num_layers=4):
        feats = self.model.net.forward(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        if num_layers is not None:
            feats = feats[:num_layers]
        return feats
