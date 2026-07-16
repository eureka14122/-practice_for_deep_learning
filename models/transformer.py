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


        assert image_size % patch_size == 0,"image为分成整数个patch"# 判断image能否被切成整数个patch
        assert embed_dim % num_heads == 0,"注意力维未被头均分"
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embed_dim,
            kernel_size = patch_size,
            stride = patch_size,
        ) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embed_dim,
            nhead = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout = 0.1,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1,1,embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1,self.num_patches + 1,embed_dim) * 0.02
        ) # 位置编码

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  # [B,1,128,128] -> [B,128，8，8]
        x = x.flatten(2) #[B,128，8，8] -> [B,128,64]
        x = x.transpose(1,2) #[B,128,64] -> [B,64,128]

        cls_tokens = self.cls_token.expand(
            B,-1,-1
        )  # [1,1,128] -> [B,1,128]

        x = torch.cat(
            (cls_tokens, x),
            dim = 1
        ) # 拼起来 —> [B,65,128]

        x = x + self.pos_embed   

        x = self.transformer(x)    
        
        cls_feat = self.norm(x[:,0])
        logits = self.head(cls_feat) # [B,128] -> [B,10]
        
        return logits
    
