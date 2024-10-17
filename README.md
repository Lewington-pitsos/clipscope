# clipscope

## usage

```python
import PIL
from clipscope import ConfiguredViT, TopKSAE

device='cpu'
filename_in_hf_repo = "725159424.pt"
sae = TopKSAE.from_pretrained(repo_id="lewington/CLIP-ViT-L-scope", filename=filename_in_hf_repo, device=device)

transformer_name='laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
locations = [(22, 'resid')]
transformer = ConfiguredViT(locations, transformer_name, device=device)

# input = PIL.Image.open("test.jpg")
# input = input.resize((224, 224)).convert("RGB")
input = PIL.Image.new("RGB", (224, 224), (0, 0, 0)) # black image for testing

activations = transformer.all_activations(input) # (1, 257, 1024)
print('activations shape', activations.shape)

output = sae(activations)

print('output keys', output.keys())

print('latent shape', output['latent'].shape) # (1, 65536)
print('reconstruction shape', output['reconstruction'].shape) # (1, 1024)
```