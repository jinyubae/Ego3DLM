import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).to(device))

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity


class ProductQuantizeEMAReset(nn.Module):
    """Product quantization: the latent is split into `codebook_num` chunks, each quantized by its own codebook."""
    def __init__(self, nb_code, code_dim, mu, codebook_num):
        super().__init__()
        assert nb_code // codebook_num, "nb_code must be divisible by codebook_num"
        assert code_dim // codebook_num, "code_dim must be divisible by codebook_num"
        
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.codebook_num = codebook_num
        
        self.codebooks = nn.ModuleList([QuantizeEMAReset(int(nb_code // codebook_num),
                                                         int(code_dim // codebook_num),
                                                         mu=mu) for i in range(codebook_num)])
    
    def quantize(self, x):
        # Chunk
        chunks = torch.chunk(x, chunks=self.codebook_num, dim=1)
        code_idx = []
        for i in range(self.codebook_num):
            code_idx_ = self.codebooks[i].quantize(chunks[i])
            code_idx.append(code_idx_)
        code_idx = torch.stack(code_idx, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        
        code_idx_chunks = torch.chunk(code_idx, chunks=self.codebook_num, dim=1)
        x = []
        for i in range(self.codebook_num):
            x_ = self.codebooks[i].dequantize(code_idx_chunks[i].squeeze(1))
            x.append(x_)
        x = torch.cat(x, dim=1)
        return x
    
    def forward(self, x):
        # Chunk
        chunks = torch.chunk(x, chunks=self.codebook_num, dim=1)
        
        x_d = []
        commit_loss = []
        perplexity = []
        for i in range(self.codebook_num):
            x_d_, commit_loss_, perplexity_ = self.codebooks[i](chunks[i])
            x_d.append(x_d_)
            commit_loss.append(commit_loss_)
            perplexity.append(perplexity_)
        x_d = torch.cat(x_d, dim=1)
        commit_loss = sum(commit_loss) / self.codebook_num
        perplexity = sum(perplexity) / self.codebook_num # average perplexity across codebooks
        
        return x_d, commit_loss, perplexity