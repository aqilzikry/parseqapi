import base64
from io import BytesIO
import torch

from flask import Flask, request, jsonify
from PIL import Image
from strhub.data.module import SceneTextDataModule

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']  # get the image file from the request
    parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
    img_transform = SceneTextDataModule.get_transform([32, 128])

    img = Image.open(img_file).convert('RGB')  # convert the file to an Image object
    img = img_transform(img).unsqueeze(0)

    logits = parseq(img)
    logits.shape

    pred = logits.softmax(-1)
    label, confidence = parseq.tokenizer.decode(pred)
    print('Decoded label = {}'.format(label[0]))
    # Do something with the image file, like process it with a model
    # and return the prediction result as a response
    return jsonify({'result': label[0]})


if __name__ == '__main__':
    app.run(debug=True)
