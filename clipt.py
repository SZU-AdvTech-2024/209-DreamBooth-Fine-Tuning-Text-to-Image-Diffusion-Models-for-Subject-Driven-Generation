import os
from PIL import Image
import numpy as np
import torch
import clip

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

your_image_folder = "./outputs2/txt2img-samples/only_bp"
your_texts = ["photo of a backpack"]

images = []
for filename in os.listdir(your_image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path = os.path.join(your_image_folder, filename)
        image = Image.open(path).convert("RGB")
        images.append(preprocess(image))

# 图像和文本预处理
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(your_texts).cuda()

# 计算特征
with torch.no_grad():
    image_features = model.encode_image(image_input).float()  # 图像的 embedding  维度512
    text_features = model.encode_text(text_tokens).float()  # 文本的 embedding

# 归一化特征
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 计算余弦相似度
similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)

# 计算平均余弦相似度
average_similarity = np.mean(similarity, axis=1)

# Average cosine similarity
print('CLIP-T score:', average_similarity)