import torch
import torch.nn as nn 

class FashionTransformer(nn.Module):
    def __init__(
        self,
        image_size = 128,
        patch_size = 16,
        in_channels = 1,
        embed_dim = 128,
        num_heads = 4,
        num_layers = 4,
        num_classes = 10,
    ):
        super().__init__()


        assert image_size % patch_size == 0 # 判断image能否被切成整数个patch

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size,
        ) # [B,1,128,128] -> [B,128，8，8]

    def forward(self, x):
        x = self.patch_embed(x)
        return x 
    


if __name__ == "__main__":
    model = FashionTransformer()

    x = torch.randn(8, 1, 128, 128)
    output = model(x)

    print(output.shape)