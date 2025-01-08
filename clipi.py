import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

# 真实图像
# real_image = preprocess(Image.open("/data/diffusers/pic/dog/02.jpg")).unsqueeze(0).to(device)

generated_image_embeddings = []

# 生成图像文件夹路径
generated_image_folder = "./outputs2/txt2img-samples/dog"
real_image_folder = "./outputs/real_dog"
# 加载并预处理生成图像
sum = 0
count = 0
for real_filename in os.listdir(real_image_folder):
    if real_filename.endswith(".png") or real_filename.endswith(".jpg"):
        real_image_path = os.path.join(real_image_folder, real_filename)
    real_image = preprocess(Image.open(real_image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        real_image_embedding = model.encode_image(real_image).cpu().numpy()

    generated_image_embeddings = []
    for filename in os.listdir(generated_image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(generated_image_folder, filename)
            generated_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            # 提取生成图像的特征向量
            with torch.no_grad():
                generated_image_embedding = model.encode_image(generated_image).cpu().numpy()
                generated_image_embeddings.append(generated_image_embedding)

# 将所有生成图像的 embedding 组合成一个数组
    generated_image_embeddings = np.vstack(generated_image_embeddings)

# # 提取真实图像的特征向量
# with torch.no_grad():
#     real_image_embedding = model.encode_image(real_image).cpu().numpy()

# 计算余弦相似度
    cosine_similarity_scores = cosine_similarity(generated_image_embeddings, real_image_embedding)

# 计算平均余弦相似度
    average_cosine_similarity = np.mean(cosine_similarity_scores)
    print("clip-i score:", average_cosine_similarity)
    sum=sum+average_cosine_similarity
    count=count+1

print("CLIP-I Score:", sum/count)