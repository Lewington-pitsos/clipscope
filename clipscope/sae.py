import torch
from huggingface_hub import hf_hub_download

class SAE(torch.nn.Module):
    @classmethod
    def _download(cls, repo_id, filename, device):
        local_file = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        checkpoint = torch.load(local_file, map_location=device)
        return checkpoint

    @classmethod
    def from_pretrained(cls, repo_id, filename, n_features=65536, d_in=1024, device='cuda'):
        checkpoint = cls._download(repo_id, filename, device)

        sae = cls(n_features=n_features, d_in=d_in, device=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
                                         
        return sae

    def __init__(self, n_features, d_in, device):
        super(SAE, self).__init__()

        self.pre_b = torch.nn.Parameter(torch.randn(d_in, device=device) * 0.01)
        self.enc = torch.nn.Parameter(torch.randn(d_in, n_features, device=device) / (2**0.5) / (d_in ** 0.5))
        self.dec = torch.nn.Parameter(self.enc.mT.clone())

        self.activation = torch.nn.ReLU()

    def _encode(self, x):
        return self.activation(x)
    
    def _decode(self, latent, dec):
        return latent, latent @ dec

    def forward_descriptive(self, x):
        encoded = self._encode(((x - self.pre_b) @ self.enc)) # (n_to_expert, expert_dim)
        latent, reconstruction = self._decode(encoded, self.dec)
        
        reconstruction = reconstruction + self.pre_b 

        _was_active = torch.max(latent, dim=0).values > 1e-3
        return {
            'reconstruction': reconstruction,
            'latent': latent,
            'experts_chosen': None,
            'expert_prop': None,
            'expert_weighting': None,
            'active_latents': _was_active,
        }

    def forward(self, x):
        recon = self.forward_descriptive(x)
        return recon

def encode_topk(pre_activation, k):
    return torch.topk(pre_activation, k=k, dim=-1) # (batch_size, k)

def eagre_decode(topk, dec):
    latent = torch.zeros((topk.values.shape[0], dec.shape[0]), dtype=dec.dtype, device=dec.device) # (batch_size, n_features)
    latent.scatter_(dim=-1, index=topk.indices, src=topk.values)

    return latent, latent @ dec
    
class TopKSAE(SAE):
    @classmethod
    def from_pretrained(cls, repo_id, filename, k=32, n_features=65536, d_in=1024, device='cuda'):
        checkpoint = cls._download(repo_id, filename, device)

        sae = cls(k=k, n_features=n_features, d_in=d_in, device=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
                                         
        return sae

    def __init__(self, k, *args, **kwargs):
        super(TopKSAE, self).__init__(*args, **kwargs)
        self.k = k
        self.activation_fn = None

    def _encode(self, x):
        return encode_topk(x, self.k)

    def _decode(self, topk, dec):
        return eagre_decode(topk, dec)
        