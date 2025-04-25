import albumentations as A
import cv2
import random

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),

    # 1. 应用一定的模糊，降低内部细节
    A.GaussianBlur(
        blur_limit=(5, 11), # 稍微增大模糊核，使内部小块更容易模糊
        p=0.6 # 提高应用概率
    ),

    # 2. 模拟中心区域的白色过曝/信息丢失
    #    使用 CoarseDropout，填充白色
    #    调整大小使其可能覆盖中心区域，但不保证完美
    A.CoarseDropout(
        max_holes=1,        # 只创建一个大区域
        max_height=128,     # 遮挡区域的最大高度 (需要根据你的图像大小调整)
        max_width=128,      # 遮挡区域的最大宽度 (需要根据你的图像大小调整)
        min_holes=1,
        min_height=64,      # 遮挡区域的最小高度
        min_width=64,       # 遮挡区域的最小宽度
        fill_value=255,     # 填充白色 (模拟过曝/信息丢失)
        mask_fill_value=None, # 如果你有掩码，这里也需要设置
        p=0.5               # 应用此效果的概率
    ),

    # 3. 强烈的亮度增加和对比度降低，进一步洗白模糊/遮挡区域
    A.RandomBrightnessContrast(
        brightness_limit=(0.4, 0.8), # 大幅提高亮度上限
        contrast_limit=(-0.5, -0.2), # 大幅降低对比度，甚至为负
        brightness_by_max=True,
        always_apply=False,
        p=0.7 # 提高应用概率
    ),

    # 4. Gamma 变换可以调整整体亮度感觉
    A.RandomGamma(
        gamma_limit=(10,30), # gamma_limit 的单位是 100*gamma，所以这里是 0.8 到 1.5
        p=0.5
    ),

    # 5. 最后应用透视变换
    A.Perspective(
        scale=(0.05, 0.1),
        pad_mode=cv2.BORDER_CONSTANT,
        # 如果原始背景是白色，填充白色可能更自然
        pad_val=random.choice([0, 255]), # 随机填充黑色或白色
        p=0.5
    ),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("test2.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 如果需要转RGB

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]

# Save the augmented image
cv2.imwrite("augmented_image.jpg", transformed_image)