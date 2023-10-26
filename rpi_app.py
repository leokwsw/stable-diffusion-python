import os
import time

from diffusers import StableDiffusionPipeline
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

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        cache_dir=os.path.join(os.path.join(os.path.dirname(__file__), "huggingface"), "hub")
    )
    pipe = pipe.to("cpu")

    image = pipe(prompt, num_inference_steps=31, width=width, height=height, num_images_per_prompt=1).images[0]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"out-{timestamp}.png")
    image.save(output_path)

    print(f"Output image saved to: {output_path}")

    return send_file(output_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)
