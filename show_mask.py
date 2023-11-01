# import numpy as np
# import torchvision.utils as vutils
# import torchvision.models as models
# import torchvision.transforms as T
# import matplotlib.pyplot as plt
# from PIL import Image
#
# imgfile = '2007_000068.jpg'
# pngfile = '2007_000068.png'
#
# mini_batch = []
# img = Image.open(imgfile)
# mask = Image.open(pngfile)
# blend_img = Image.blend(img, mask, alpha=0.5)
# blend_img_tensor = T.functional.to_tensor(blend_img)
# mini_batch.append(blend_img_tensor)
# grid_img = vutils.make_grid(mini_batch,padding=3,pad_value=1)
# plt.axis('off')
# plt.imshow(grid_img.permute(1,2,0))
# plt.show()
#
# # img = cv2.imread(imgfile, 1)
# # mask = cv2.imread(pngfile, 0)
#
# # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
# #
# # img = img[:, :, ::-1]
# # img[..., 2] = np.where(mask == 1, 255, img[..., 2])
#
# plt.imshow(img)
# plt.show()
# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt

# image1 原图
# image2 分割图
image1 = Image.open("2008_001439.jpg")
image2 = Image.open("2008_001439.png")

image1 = image1.convert('RGBA')
image2 = image2.convert('RGBA')

# 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
image = Image.blend(image1, image2, 0.7)
image.save("test.png")
image.show()
