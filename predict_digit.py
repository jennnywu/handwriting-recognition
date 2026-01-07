import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

model = tf.keras.models.load_model("handwritten.keras")

def prep_mnist_like(bgr_or_gray_img):
    if len(bgr_or_gray_img.shape) == 3:
        img = cv2.cvtColor(bgr_or_gray_img, cv2.COLOR_BGR2GRAY)
    else:
        img = bgr_or_gray_img.copy()

    if img.mean() > 127:
        img = 255 - img

    img = cv2.GaussianBlur(img, (3, 3), 0)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("No digit pixels found")

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left

    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )

    x = img.astype("float32") / 255.0
    x = x[None, ..., None]

    return x, img

digit = 0

while os.path.isfile(f"digits/test{digit}.png"):
    path = f"digits/test{digit}.png"
    
    try:
        raw = cv2.imread(path)
        if raw is None:
            raise FileNotFoundError(path)

        x, vis = prep_mnist_like(raw)

        prediction = model.predict(x, verbose=0)[0]
        pred_class = int(np.argmax(prediction))

        print(f"Prediction for test{digit}.png: {pred_class}")

        plt.imshow(vis, cmap=plt.cm.binary)
        plt.title(f"test{digit}.png: {pred_class}")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Could not process test{digit}.png: {e}")

    finally:
        digit += 1