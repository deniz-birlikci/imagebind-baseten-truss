# ImageBind - API Endpoint Deployment (with Baseten)
Implementation of Imagebind model with Baseten's Truss architecture. Allows anyone to host ImageBind model as an endpoint in the web using Baseten.

## How to set up Baseten
You can follow along the setup.ipynb notebook as well. Here are the core steps:
```python
import baseten
import truss

baseten.login('BASETEN_API_KEY')

my_packaged_model = truss.load('imagebind-baseten-truss/imagebind_serializable_truss/')

baseten.deploy(
    my_packaged_model,
    model_name='imageBind',
    # publish=True, # Uncomment to skip the draft stage
)
```
After you deploy, the console output will include a model version id. You could also get this from baseten's website under your deployed models. We take this version id and use it to retrieve the model endpoint from Baseten.
```python
# Retrieve the uploaded model endpoint from baseten
model = baseten.deployed_model_version_id('<model_version_id>')
```

## How to run inference
Recall that we had recieved our model version ID from our deployment. We use that ID again to retrieve the model endpoint.
```python
# Retrieve the uploaded model endpoint from baseten
model = baseten.deployed_model_version_id('<model_version_id>')
```
Then, we have a suite of helper functions to be able to convert images or audio files into a JSON-Serializable format. This is necessary because Baseten only takes JSON-Serializable format as input. ImageBind is a multimodal modal that could take text, vision, or audio modalities, so converting each to a base64 string makes it straightforward.
```python
import base64
from PIL import Image
from pydub import AudioSegment
import io

def image_paths_to_base64_strings(image_paths: list[str]) -> list[str]:
    """
    Convert a list of image paths to a list of Base64 encoded strings.

    Parameters:
    - image_paths (list[str]): List of paths to the images.

    Returns:
    list[str]: List containing Base64 encoded strings of images.
    """
    image_base64_strings = []
    for path in image_paths:
        with Image.open(path) as img:
            binary_io = io.BytesIO()
            img.save(binary_io, format="PNG")  # You can change the format if needed
            base64_encoded = base64.b64encode(binary_io.getvalue()).decode('utf-8')
            image_base64_strings.append(base64_encoded)
    return image_base64_strings

def audio_paths_to_base64_strings(audio_paths: list[str]) -> list[str]:
    """
    Convert a list of audio paths to a list of Base64 encoded strings.

    Parameters:
    - audio_paths (list[str]): List of paths to the audio files.

    Returns:
    list[str]: List containing Base64 encoded strings of audio files.
    """

    audio_base64_strings = []
    for path in audio_paths:
        audio = AudioSegment.from_file(path)
        binary_io = io.BytesIO()
        audio.export(binary_io, format="wav")  # You can change the format if needed
        base64_encoded = base64.b64encode(binary_io.getvalue()).decode('utf-8')
        audio_base64_strings.append(base64_encoded)
    return audio_base64_strings
```
Then finally, you can get the embeddings from ImageBind using the following structure.
```python
text_list = ["cat meowing"]
image_paths = ["testing_data/cat.jpeg"]
audio_paths = ["testing_data/meow.mp3"]

json_serializable_input = {
    "text" : text_list,
    "vision" : image_paths_to_base64_strings(image_paths),
    "audio" : audio_paths_to_base64_strings(audio_paths)
}

model.predict(json_serializable_input)
```

Have fun!