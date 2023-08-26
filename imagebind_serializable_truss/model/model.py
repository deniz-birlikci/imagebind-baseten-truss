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
import base64
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
        self._bpe_path = os.path.join(self._data_dir, "bpe_simple_vocab_16e6.txt.gz")

    def load(self):
        # Load model here and assign to self._model.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # This is a imagebind implementation that is able to take
        # JSON-Serializable inputs. This is critical for hosting on Baseten.
        # https://github.com/deniz-birlikci/JSON-Serializable-ImageBind
        model = imagebind_model.imagebind_huge_serializable(device=device, 
                                                            pretrained=True, 
                                                            bpe_path=self._bpe_path)
        model.eval()
        model.to(device)

        self._model = model
        self._device = device
        
    def _is_valid_base64_string(self, s: str) -> bool:
        """Checks if a string is a valid Base64 encoded string."""
        try:
            base64.b64decode(s, validate=True)
            return True
        except Exception:
            return False
        
    def validate_serializable_input(self, model_input: dict) -> None:
        """
        Validates the format of the provided model_input dictionary.

        The dictionary should be JSON-serializable and can contain any combination of the following keys:
        - `text`: representing a list of strings.
        - `vision`: representing a list of Base64 encoded strings (images).
        - `audio`: representing a list of Base64 encoded strings (audio clips).

        Parameters:
        - model_input (dict): The input data to be validated.

        Raises:
        Exception: If the model_input is not in the correct format or contains invalid Base64 strings.
        """

        if not isinstance(model_input, dict):
            raise Exception("model_input must be a dictionary.")

        valid_keys = [ModalityType.TEXT, ModalityType.VISION, ModalityType.AUDIO]

        # Check for invalid keys
        for key in model_input.keys():
            if key not in valid_keys:
                raise Exception(f"Invalid key '{key}' in model_input. Expected keys are: {valid_keys}")

        # Check types of values
        if ModalityType.TEXT in model_input and not all(isinstance(item, str) for item in model_input[ModalityType.TEXT]):
            raise Exception("Values for the 'text' key should be a list of strings.")

        if ModalityType.VISION in model_input:
            if not all(isinstance(item, str) for item in model_input[ModalityType.VISION]):
                raise Exception("Values for the 'vision' key should be strings.")
            if not all(self._is_valid_base64_string(item) for item in model_input[ModalityType.VISION]):
                raise Exception("All values for the 'vision' key should be valid Base64 encoded strings.")

        if ModalityType.AUDIO in model_input:
            if not all(isinstance(item, str) for item in model_input[ModalityType.AUDIO]):
                raise Exception("Values for the 'audio' key should be strings.")
            if not all(self._is_valid_base64_string(item) for item in model_input[ModalityType.AUDIO]):
                raise Exception("All values for the 'audio' key should be valid Base64 encoded strings.")

    def predict(self, model_input):
        """
        Generate embeddings for the given model input.

        The `model_input` should be a JSON-serializable dictionary. It can contain any combination of the following keys:
        - `text`: representing a list of strings.
        - `vision`: representing a list of binaries (images).
        - `audio`: representing a list of binaries (audio clips).

        It's not mandatory for all keys to be present. The function will only process the modalities present in the model_input.

        This function preprocesses the input, gets the embeddings using the model, and then returns a dictionary containing the embeddings mapped to their respective ModalityType.

        Parameters:
        - model_input (dict): The input data in a dictionary format. Depending on the modalities you wish to process, it can have any of the following structures:
            {
                "text": [<list of strings>],
                ...
            }
            {
                "vision": [<list of image binaries>],
                ...
            }
            {
                "audio": [<list of audio binaries>],
                ...
            }
            ... or any combination thereof.

        Returns:
        dict: A dictionary with keys mapped to ModalityType enum and values as lists containing the embeddings of the respective modalities.

        Raises:
        ValueError: If the model_input is not of type dictionary.

        Example:
        Given:
            model_input = {
                "text": ["Hello, world!"],
                "audio": [<binary audio data>]
            }

        Returns:
            {
                ModalityType.TEXT: [<list of numerical values representing the text embedding>],
                ModalityType.AUDIO: [<list of numerical values representing the audio embedding>]
            }
        """
        # Make sure that the model input is valid. If not, this raises
        # an exception
        self.validate_serializable_input(model_input)
        
        with torch.no_grad():
            embeddings = self._model(model_input)

        output = {}
        for modality_type, embedding in embeddings.items():
            output[modality_type] = embedding.tolist()
            
        del model_input
        
        return output
    
