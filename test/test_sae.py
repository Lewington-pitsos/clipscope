import PIL
from clipscope import ConfiguredViT, TopKSAE

def test_sae():


    device='cpu'
    filename_in_hf_repo = "725159424.pt"
    sae = TopKSAE.from_pretrained(repo_id="lewington/CLIP-ViT-L-scope", filename=filename_in_hf_repo, device=device)

    transformer_name='laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
    locations = [(22, 'resid')]
    transformer = ConfiguredViT(locations, transformer_name, device=device)

    input = PIL.Image.new("RGB", (224, 224), (0, 0, 0)) # black image for testing

    activations = transformer.all_activations(input)[locations[0]] # (1, 257, 1024)
    assert activations.shape == (1, 257, 1024)

    activations = activations[:, 0, :] # just the cls token

    output = sae.forward_verbose(activations)

    print('output keys', output.keys())
    assert 'latent' in output
    assert 'reconstruction' in output

    assert output['latent'].shape == (1, 65536)
    assert output['reconstruction'].shape == (1, 1024)

