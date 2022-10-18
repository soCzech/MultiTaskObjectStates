import sys
import torch
sys.path.append("./CLIP")
from clip.model import VisionTransformer


class ClassifierHeads(torch.nn.Module):

    def __init__(self, layers, n_classes=1):
        super(ClassifierHeads, self).__init__()

        self.state = torch.nn.ModuleDict({
            f"l{i:d}": torch.nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        })
        self.action = torch.nn.ModuleDict({
            f"l{i:d}": torch.nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        })

        self.state_layer = torch.nn.Linear(layers[-1], 2 * n_classes + 1, bias=True)
        self.action_layer = torch.nn.Linear(layers[-1], 1 * n_classes + 1, bias=True)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.state)):
            x = self.state[f"l{i:d}"](x)
            x = torch.relu(x)
        state = self.state_layer(x)

        x = inputs
        for i in range(len(self.action)):
            x = self.action[f"l{i:d}"](x)
            x = torch.relu(x)
        action = self.action_layer(x)

        return {"state": state, "action": action}


class ClipClassifier(torch.nn.Module):

    def __init__(self, hidden_mlp_layers, params, n_classes=1, train_backbone=True):
        super(ClipClassifier, self).__init__()

        self.args = params
        self.n_classes = n_classes
        self.hidden_mlp_layers = hidden_mlp_layers

        if "visual.conv1.weight" in params:
            vision_width = params["visual.conv1.weight"].shape[0]
            vision_heads = vision_width // 64
            vision_layers = len([k for k in params.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = params["visual.conv1.weight"].shape[-1]
            grid_size = round((params["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
            self.args = dict(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=None
            )

        self.backbone = VisionTransformer(**self.args)

        if "visual.conv1.weight" in params:
            self.backbone.load_state_dict({
                k[len("visual."):]: v for k, v in params.items()
                if k.startswith("visual.") and k != "visual.proj"
            })

        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        self.heads = ClassifierHeads([self.args["width"]] + hidden_mlp_layers, n_classes)

    @staticmethod
    def preprocess(imgs):
        # CLIP image preprocessing
        imgs = imgs.permute((0, 3, 1, 2)).float().div_(255)
        mean = torch.as_tensor((0.48145466, 0.4578275, 0.40821073),
                               dtype=torch.float32, device=imgs.device).view(1, -1, 1, 1)
        std = torch.as_tensor((0.26862954, 0.26130258, 0.27577711),
                              dtype=torch.float32, device=imgs.device).view(1, -1, 1, 1)
        return imgs.sub_(mean).div_(std)

    def forward(self, inputs):
        x = self.preprocess(inputs)
        x = self.backbone(x)
        x = self.heads(x)
        return x
