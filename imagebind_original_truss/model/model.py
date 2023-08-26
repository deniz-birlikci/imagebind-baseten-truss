"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""
import os
import imagebind
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)

        self._model = model
        self._device = device

    def predict(self, model_input):
        # Run model inference here
        if not isinstance(model_input, list) and isinstance(model_input[0], str):
            raise ValueError("Expecting List[str]")
        
        bpe_path = os.path.join(self._data_dir, "bpe_simple_vocab_16e6.txt.gz")
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(model_input, self._device, bpe_path),
        }
        
        with torch.no_grad():
            embedding = self._model(inputs)
            
        vector = embedding['text'].tolist()
        
        del inputs
        torch.cuda.empty_cache()
        
        return vector
    
