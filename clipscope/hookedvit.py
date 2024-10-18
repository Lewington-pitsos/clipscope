import torch
from transformers import CLIPProcessor, CLIPModel
from contextlib import contextmanager
from functools import partial

# a lot of this code is copy-pasted from https://github.com/HugoFry/mats_sae_training_for_ViTs

TRANSFORMER_NAME = 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'

class Hook():
    def __init__(self, block_layer: int, module_name: str, hook_fn, return_module_output = True):
        self.path_dict = {
            'resid': '',
        }
        assert module_name in self.path_dict.keys(), f'Module name \'{module_name}\' not recognised.'
        self.return_module_output = return_module_output
        self.function = self.get_full_hook_fn(hook_fn)
        self.attr_path = self.get_attr_path(block_layer, module_name)

    def get_full_hook_fn(self, hook_fn):

        def full_hook_fn(module, module_input, module_output):
            hook_fn_output = hook_fn(module_output[0])
            if self.return_module_output:
                return module_output
            else:
                return hook_fn_output # Inexplicably, the module output is not a tensor of activaitons but a tuple (tensor,)...??

        return full_hook_fn

    def get_attr_path(self, block_layer: int, module_name: str) -> str:
        attr_path = f'vision_model.encoder.layers[{block_layer}]'
        attr_path += self.path_dict[module_name]
        return attr_path

    def get_module(self, model):
        return self.get_nested_attr(model, self.attr_path)

    def get_nested_attr(self, model, attr_path):
        """
        Gets a nested attribute from an object using a dot-separated path.
        """
        module = model
        attributes = attr_path.split(".")
        for attr in attributes:
            if '[' in attr:
                # Split at '[' and remove the trailing ']' from the index
                attr_name, index = attr[:-1].split('[')
                module = getattr(module, attr_name)[int(index)]
            else:
                module = getattr(module, attr)
        return module


class HookedViT():
    def __init__(self, model_name: str = TRANSFORMER_NAME, device = 'cuda'):
        model, processor = self.get_ViT(model_name)
        self.model = model.to(device)
        self.processor = processor

    def get_ViT(self, model_name):
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    def run_with_cache(self, list_of_hook_locations, *args, **kwargs):
        cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)
        return output, cache_dict

    def get_caching_hooks(self, list_of_hook_locations):
        """
        Note that the cache dictionary is index by the tuple (block_layer, module_name).
        """
        cache_dict = {}
        list_of_hooks=[]
        def save_activations(name, activations):
            cache_dict[name] = activations.detach()
        for (block_layer, module_name) in list_of_hook_locations:
            hook_fn = partial(save_activations, (block_layer, module_name))
            hook = Hook(block_layer, module_name, hook_fn)
            list_of_hooks.append(hook)
        return cache_dict, list_of_hooks

    @torch.no_grad
    def run_with_hooks(self, list_of_hooks, *args, **kwargs):
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                return hooked_model(*args, **kwargs)
    
    @contextmanager
    def hooks(self, hooks):
        """

        This is a context manager for running a model with hooks. The funciton adds 
        forward hooks to the model, and then returns the hooked model to be run with 
        a foward pass. The funciton then cleans up by removing any hooks.

        Args:

          model VisionTransformer: The ViT that you want to run with the forward hook

          hooks List[Tuple[str, Callable]]: A list of forward hooks to add to the model. 
            Each hook is a tuple of the module name, and the hook funciton.

        """
        hook_handles = []
        try:
            for hook in hooks:
                module = hook.get_module(self.model)
                handle = module.register_forward_hook(hook.function)
                hook_handles.append(handle)
            yield self.model
        finally:
            for handle in hook_handles:
                handle.remove()
                
    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()

class ConfiguredViT(HookedViT):
    def __init__(self, hook_locations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._hook_locations = hook_locations

    def all_activations(self, batch):
        inputs = self.processor(images=batch, text = "", return_tensors="pt", padding = True).to(self.model.device)

        _, cache_dict = self.run_with_cache(
            self._hook_locations,
            **inputs,
        )

        return cache_dict

    def cls_activations(self, batch):
        cache_dict = self.all_activations(batch)

        for k, v in cache_dict.items():
            cache_dict[k] = v[:,0]

        return cache_dict
