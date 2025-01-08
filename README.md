# Dreambooth on Stable Diffusion

这是Google的[Dreambooth](https://arxiv.org/abs/2208.12242) 基于 [Stable Diffusion](https://github.com/CompVis/stable-diffusion)实现的一个项目。 最初的 Dreambooth 基于 Imagen 文本到图像模型。然而，Imagen 的模型和预训练权重都不可用。为了让更多人能通过一些示例对文本到图像模型进行微调，我在 Stable diffusion 上实现了 Dreambooth 的idea。

这个代码库主要是基于 [Textual Inversion](https://github.com/rinongal/textual_inversion). 不同的是，Textual Inversion 只优化了单词嵌入，而 Dreambooth 则对整个扩散模型进行了微调。

### Preparation
首先，根据项目中的environment.yaml文件配置ldm环境。

需要对stable diffusion进行微调，根据这个[说明](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1)获取预训练模型. 权重参数在 [HuggingFace](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)下载。在这里提供的链接是我使用的 ```sd-v1-4-full-ema.ckpt```。

根据Dreambooth 的微调算法流程，还需要创建一组用于正则化的图像。生成正则化图像的文本提示可以是 ```photo of a <class>``` ，其中 ```<class>``` 是描述对象类别的单词，例如 ```dog```。执行命令：

```
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /path/sd/sd-v1-4-full-ema.ckpt --prompt "a photo of a <class>" 
```

这里的```n_sample```为8表示为正则化生成了 8 幅图像，但更多的正则化图像可能会带来更强的正则化效果和更好的可编辑性。然后，将每个生成的图像放在用于正则化的文件中，eg：```/root/to/regularization/images```。


### Training
执行以下命令，

```
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml 
                -t 
                --actual_resume /path/sd/sd-v1-4-full-ema.ckpt  
                -n <job name> 
                --gpus 0, 
                --data_root /root/to/training/images 
                --reg_data_root /root/to/regularization/images 
                --class_word <xxx>
```

关于更加详细的config，见 ```configs/stable-diffusion/v1-finetune_unfrozen.yaml```. 默认的学习率为 ```1.0e-6``` 。参数```reg_weight```与 Dreambooth 论文中的正则化权重相对应，默认设置为```1.0```。

Dreambooth 需要一个占位词```[V]```，称为标识符（identifier）。这个标识符必须是词汇中相对罕见的词。原论文通过在 T5-XXL 标记器中使用稀有词来解决这个问题。为了简单起见，在这里只使用了一个随机词“`sks`”，并对其进行了硬编码。具体还可以在这个[文件](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion/blob/main/ldm/data/personalized.py#L10)里面修改。

Training会执行800个step, 并且产生两个checkpoints，保存在 ```./logs/<job_name>/checkpoints```。我使用几张小猫的图片针对这个预训练模型进行微调训练后，得到的checkpoints保存在```.\logs\images2025-01-01T11-55-26_cat\checkpoints\last.ckpt```，由于这个保存文件太大，我将整个[logs文件](https://drive.google.com/file/d/1q5VDLX1HBhYnJzlfTlMjb3Y7fMd2Svvj/view?usp=sharing)进行压缩打包，只需要解压后把这个文件放在根目录下即可，然后就可以直接在generation中的```--ckpt```中直接使用这个ckpt进行生成。

### Generation
训练完成后，执行命令

```
python scripts/stable_txt2img.py --ddim_eta 0.0 
                                 --n_samples 8 
                                 --n_iter 1 
                                 --scale 10.0 
                                 --ddim_steps 100  
                                 --ckpt /logs/to/saved/checkpoint/from/training
                                 --prompt "photo of a sks <class>" 
```

其中，```sks```是标识符，如果要更改标识符，则应根据自己的选择进行替换；```<class>```是用于训练的类。

