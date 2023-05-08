import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform([32, 128])

# print(dir(parseq))

img = Image.open('image/test4.jpg').convert('RGB')
img = img_transform(img).unsqueeze(0)

logits = parseq(img)
logits.shape

pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))