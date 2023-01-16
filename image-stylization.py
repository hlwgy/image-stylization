# %% 
import os 
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
"""
https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1
对应下载连接为
https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/1.tar.gz
https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/2.tar.gz
即替换tfhub.dev为storage.googleapis.com/tfhub-modules，并且在末尾加上后缀.tar.gz然后将文件解压到自己喜欢的路径
"""
hub_model1 = hub.load('image-stylization-v1-256_1')
hub_model2 = hub.load('image-stylization-v1-256_2')
# %% 
# tensor转image图片
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor_arr = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor_arr)>3:
    assert tensor_arr.shape[0] == 1
    img_arr = tensor_arr[0]
  return img_arr

# 根据路径加载图片，改造成512尺寸的标准
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

# 进行转化
def stylization(content_path, style_path, mode = "m2"):
  content_image = load_img(content_path)
  style_image = load_img(style_path)
  if mode == "m1":
    stylized_images = hub_model1(tf.constant(content_image), tf.constant(style_image))
    return tensor_to_image(stylized_images[0])
  if mode == "m2":
    stylized_images = hub_model2(tf.constant(content_image), tf.constant(style_image))
    return tensor_to_image(stylized_images[0])
# %%
def main():

  mode = "m2"
  dir_style = "style2"
  styles = [f for f in os.listdir(dir_style)]
  content_dir = "content"
  for style in styles:
    # if style != "bg.png":
    #   continue
    f_names = [f for f in os.listdir(content_dir)]
    for i, n in enumerate(f_names):
      if n.startswith("_"):
        continue
      print(i, "====>", n)
      f_path = os.path.join(content_dir , n)
      img_arr = stylization(f_path, dir_style+"\\"+style, mode)
      image_pil = PIL.Image.fromarray(img_arr)
      n_path = os.path.join(content_dir, "_"+style+"+"+mode+"-"+n)
      image_pil.save(n_path)
  print("all completed!")

main()
# %%
