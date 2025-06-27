from openvino.runtime import Core
from transformers import BlipProcessor
from PIL import Image
import numpy as np

# Paths to your IR files
model_xml = "blip_ov/blip_caption.xml"
model_bin = "blip_ov/blip_caption.bin"

# Load OpenVINO model
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Load tokenizer/processor (from HF)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# Load and preprocess image
image = Image.open("1.png").convert("RGB")
inputs = processor(images=image, text=["a photo of"], return_tensors="np")

input_keys = list(compiled_model.inputs)
input_dict = {
    input_keys[0].get_any_name(): inputs["input_ids"],
    input_keys[1].get_any_name(): inputs["attention_mask"],
    input_keys[2].get_any_name(): inputs["pixel_values"],
}

# Run inference
outputs = compiled_model(input_dict)
output_ids = np.argmax(list(outputs.values())[0], axis=-1)

# Decode caption
caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("🖼️ Caption:", caption)
