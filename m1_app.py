import os
import torch
import time
from diffusers import DiffusionPipeline
from flask import Flask, request, send_file, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    prompt = data['prompt']

    if "width" in data:
        width = data['width']
    else:
        width = 512

    if "height" in data:
        height = data['height']
    else:
        height = 512

    if "steps" in data:
        steps = data['steps']
    else:
        steps = 8

    if "seed" in data:
        seed = data['seed']
    else:
        seed = None

    predictor = Predictor()

    output_path = predictor.predict(prompt, width, height, steps, seed)
    print(f"Output image saved to: {output_path}")

    # return {'path': output_path}
    return send_file(output_path, mimetype='image/png')


class Predictor:
    def __init__(self):
        self.pipe = self._load_model()

    def _load_model(self):
        model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
            cache_dir=os.path.join(os.path.join(os.path.dirname(__file__), "huggingface"), "hub")
        )
        model.to(torch_device="cpu", torch_dtype=torch.float32).to('mps:0')
        return model

    def predict(self, prompt: str, width: int, height: int, steps: int, seed: int = None) -> str:
        seed = seed or int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        result = self.pipe(
            prompt=prompt, width=width, height=height,
            guidance_scale=8.0, num_inference_steps=steps,
            num_images_per_prompt=1, lcm_origin_steps=50,
            output_type="pil"
        ).images[0]

        return self._save_result(result)

    def _save_result(self, result):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"out-{timestamp}.png")
        result.save(output_path)
        return output_path


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)
