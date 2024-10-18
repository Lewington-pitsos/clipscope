import PIL
from clipscope import ConfiguredViT, TopKSAE

def test_sae():
    device='cpu'
    filename_in_hf_repo = "22_resid/1200013184.pt"
    sae = TopKSAE.from_pretrained(checkpoint=filename_in_hf_repo, device=device)

    locations = [(22, 'resid')]
    transformer = ConfiguredViT(locations, device=device)

    input = PIL.Image.new("RGB", (224, 224), (0, 0, 0)) # black image for testing

    activations = transformer.all_activations(input)[locations[0]] # (1, 257, 1024)
    assert activations.shape == (1, 257, 1024)

    activations = activations[:, 0, :] # just the cls token

    output = sae.forward_verbose(activations)

    assert 'latent' in output
    assert 'reconstruction' in output

    assert output['latent'].shape == (1, 65536)
    assert output['reconstruction'].shape == (1, 1024)

