to implement the model pytorch code is as follows:

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import os
import matplotlib.pyplot as plt
from transformers import BartTokenizer
from google.colab import drive

drive.mount('/content/drive')

# Load tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
# ---- Core transformer improvements ----
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim*mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*mult, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_mult, dropout=dropout)
    def forward(self, x):
        # x: [seq_len, B, D]
        xa = self.norm1(x)
        attn_out, _ = self.attn(xa, xa, xa)
        x = x + attn_out
        xf = self.norm2(x)
        x = x + self.ff(xf)
        return x
class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=mlp_mult, dropout=dropout)
    def forward(self, x, memory, tgt_mask=None):
        # Self-attention
        xs = self.norm1(x)
        self_out, _ = self.self_attn(xs, xs, xs, attn_mask=tgt_mask)
        x = x + self_out
        # Cross-attention
        xc = self.norm2(x)
        cross_out, _ = self.cross_attn(xc, memory, memory)
        x = x + cross_out
        # FFN
        xf = self.norm3(x)
        x = x + self.ff(xf)
        return x
# ---- Encoder with enhanced transformer ----
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=768, nhead=8, num_layers=4, freeze_backbone=True):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad=False
        self.non_local = nn.Conv2d(2048, 2048, 1)  # simple channel mixer
        self.proj = nn.Linear(2048, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, nhead) for _ in range(num_layers)
        ])
        self.graph = nn.Linear(embed_dim, embed_dim)  # lightweight graph residual
        self.global_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        feat = self.backbone(x)        # [B,2048,H,W]
        feat = self.non_local(feat)    # channel mix
        B,C,H,W = feat.shape
        tokens = feat.view(B,C,H*W).permute(0,2,1)  # [B,N,C]
        tokens = self.proj(tokens)               # [B,N,D]
        tokens = tokens + self.pos_embed[:,:H*W,:]
        tokens = tokens.permute(1,0,2)           # [N,B,D]
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = tokens + self.graph(tokens)
        global_tokens = self.global_token.expand(1, B, -1)  # [1,B,D]
        tokens = torch.cat([tokens, global_tokens], dim=0)  # [50,B,D]
        return tokens

# ---- Decoder with improved transformer blocks ----
class TransformerTextDecoder(nn.Module):
    def __init__(self, vocab_size, max_len=300, embed_dim=768, nhead=8, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.layers = nn.ModuleList([
            HierarchicalDecoderBlock(embed_dim, nhead) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_ids, memory):
        seq_len = tgt_ids.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt_ids.device)
        x = self.embed(tgt_ids) + self.pos_embed[:, :seq_len, :]
        x = x.permute(1,0,2)  # [S,B,D]
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask)
        x = self.norm(x).permute(1,0,2)  # [B,S,D]
        return self.fc_out(x)


class HierarchicalDecoderBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_mult=4, dropout=0.1):
        super().__init__()
        # Inherit components from TransformerDecoderBlock
        self.self_attn_block = TransformerDecoderBlock(dim, nhead, mlp_mult, dropout)
        
        # Hierarchical cross-attentions
        self.local_cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.global_cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout)
        self.norm_local = nn.LayerNorm(dim)
        self.norm_global = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        # Phase 1: Standard self-attention + FFN
        x = self.self_attn_block(x, memory, tgt_mask)
        
        # Phase 2: Hierarchical cross-attention
        # Local attention (first 49 image tokens)
        x_local = self.norm_local(x)
        local_out, _ = self.local_cross_attn(
            query=x_local,
            key=memory[:49],  # First 49 tokens = local features
            value=memory[:49]
        )
        x = x + self.dropout(local_out)
        
        # Global attention (remaining tokens)
        x_global = self.norm_global(x)
        global_out, _ = self.global_cross_attn(
            query=x_global,
            key=memory[49:],  # Assume last token is global context
            value=memory[49:]
        )
        x = x + self.dropout(global_out)
        
        return x
# ---- Final model ----
class AdvancedCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=768):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.decoder = TransformerTextDecoder(vocab_size, embed_dim=embed_dim)

    def forward(self, images, captions):
        memory = self.encoder(images)
        return self.decoder(captions, memory)

    def generate(self, images, tokenizer, max_len=300):
        self.eval()
        with torch.no_grad():
            memory = self.encoder(images)
            B = images.size(0)
            tokens = torch.full((B,1), tokenizer.bos_token_id, dtype=torch.long, device=images.device)
            for _ in range(max_len):
                out = self.decoder(tokens, memory)
                next_tok = out[:, -1:].argmax(-1)
                tokens = torch.cat([tokens, next_tok], dim=1)
                if (next_tok == tokenizer.eos_token_id).all(): break
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)
# ---- Inference Script ----
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = AdvancedCaptionModel(tokenizer.vocab_size).to(device)

    # Load trained weights
    ckpt_path = '/content/drive/MyDrive/advanced_HiCap_10.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    def load_image(path):
        img = Image.open(path).convert('RGB')
        return preprocess(img).unsqueeze(0).to(device)

    # Caption generation
    def generate_caption(image_tensor):
        with torch.no_grad():
            return model.generate(image_tensor, tokenizer, max_len=30)

    # Example inference
    img_path = '/content/drive/MyDrive/Flickr30k/Images/88279365.jpg'
    img_tensor = load_image(img_path)
    caption = generate_caption(img_tensor)
    print('Generated Caption:', caption[0])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
