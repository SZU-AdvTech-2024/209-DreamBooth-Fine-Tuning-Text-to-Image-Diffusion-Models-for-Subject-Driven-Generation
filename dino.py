from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch.nn as nn
import torch
import os

# 定义环境
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 加载 dinov2 模型
model_folder = '/data/diffusers/model/dinov2-base'
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
real_image_folder = './outputs/real_dog'
# 加载并处理真实图像




# 初始化相似度计算
cos = nn.CosineSimilarity(dim=0)

# 设置生成图像文件夹路径
generated_image_folder = './outputs2/txt2img-samples/dog'

# 初始化相似度总和和计数器
total_similarity = 0
count = 0

for real_filename  in os.listdir(real_image_folder):
    if real_filename.endswith(".png") or real_filename.endswith(".jpg"):
        real_image_path = os.path.join(real_image_folder, real_filename)

        real_image = Image.open(real_image_path)
        with torch.no_grad():
            inputs1 = processor(images=real_image, return_tensors="pt").to(device)
            outputs1 = model(**inputs1)
            image_features1 = outputs1.last_hidden_state
            image_features1 = image_features1.mean(dim=1)
# 遍历生成图像文件夹并计算相似度
        for filename in os.listdir(generated_image_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(generated_image_folder, filename)

                # 加载生成图像并处理
                generated_image = Image.open(image_path)
                with torch.no_grad():
                    inputs2 = processor(images=generated_image, return_tensors="pt").to(device)
                    outputs2 = model(**inputs2)
                    image_features2 = outputs2.last_hidden_state
                    image_features2 = image_features2.mean(dim=1)

                # 计算相似度
                sim = cos(image_features1[0], image_features2[0]).item()
                sim = (sim + 1) / 2  # 将相似度值归一化到 [0, 1] 范围

                # 累加相似度值和计数
                total_similarity += sim
                count += 1
                # print(f'real_image和{filename}的相似度值: {sim}')

# 计算平均相似度并输出
if count > 0:
    average_similarity = total_similarity / count
    print(f'DINO score: {average_similarity}')
else:
    print('没有找到生成图像。')