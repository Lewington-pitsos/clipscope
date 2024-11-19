# clipscope

[Github Repo](https://github.com/Lewington-pitsos/clipscope)

## usage

```python
import PIL
from clipscope import ConfiguredViT, TopKSAE

device='cpu'
filename_in_hf_repo = "22_resid/1200013184.pt"
sae = TopKSAE.from_pretrained(checkpoint=filename_in_hf_repo, device=device)

locations = [(22, 'resid')]
transformer = ConfiguredViT(locations, device=device)

input = PIL.Image.new("RGB", (224, 224), (0, 0, 0)) # black image for testing

activations = transformer.all_activations(input)[locations[0]] # (1, 257, 1024)
assert activations.shape == (1, 257, 1024)

activations = activations[:, 0] # just the cls token
# alternatively flatten the activations
# activations = activations.flatten(1)

print('activations shape', activations.shape)

output = sae.forward_verbose(activations)

print('output keys', output.keys())

print('latent shape', output['latent'].shape) # (1, 65536)
print('reconstruction shape', output['reconstruction'].shape) # (1, 1024)
```