import numpy as np
from PIL import Image
import cv2


def preprocess_image(image, watermark_type):
    image_type: str = ''
    preprocessed_mask_image = np.array([])
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image_h = image.shape[0]
    image_w = image.shape[1]
    aspectRatioImage = image_w / image_h
    print("image size: {}".format(image.shape))

    if image_w > image_h:
        image_type = "landscape"
    elif image_w == image_h:
        image_type = "landscape"
    else:
        image_type = "potrait"

    mask_image = Image.open(
        "utils/{}/{}/mask.png".format(watermark_type, image_type))
    if mask_image.mode != "RGB":
        mask_image = mask_image.convert("RGB")
    mask_image = np.array(mask_image)
    print("mask image size: {}".format(mask_image.shape))

    aspectRatioMaskImage = mask_image.shape[1] / mask_image.shape[0]
    upperBoundAspectRatio = 1.05 * aspectRatioMaskImage
    lowerBoundAspectRatio = 0.95 * aspectRatioMaskImage

    if aspectRatioImage >= lowerBoundAspectRatio and aspectRatioImage <= upperBoundAspectRatio:
        preprocessed_mask_image = cv2.resize(mask_image, (image_w, image_h))
        print(preprocessed_mask_image.shape)
    else:
        print("Image size not supported!!!")

    if (preprocessed_mask_image.shape != (0,)):
        assert image.shape == preprocessed_mask_image
        grid = 8
        image = image[:image_h//grid*grid, :image_w//grid*grid, :]
        preprocessed_mask_image = preprocessed_mask_image[:image_h //
                                                          grid*grid, :image_w//grid*grid, :]
        image = np.expand_dims(image, 0)
        preprocessed_mask_image = np.expand_dims(preprocessed_mask_image, 0)
        input_image = np.concatenate([image, preprocessed_mask_image], axis=2)
        return input_image

    else:
        return preprocessed_mask_image
