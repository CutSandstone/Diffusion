import torch
import cv2
import math
import matplotlib.pyplot as plt
def addNoise(img,beta_t):
    return torch.normal(math.sqrt(1-beta_t)*img,math.sqrt(beta_t))
def genNoise(img,n):
    def f(t):
        s = 0.008
        return math.cos((t/n+s)/(1+s)*math.pi/2)**2
    image_list = [img]
    for t in range(1,n):
        b_t = 1-f(t)/f(0)
        img = addNoise(img,b_t)
        image_list.append(img)
    return image_list
if __name__ == '__main__':
    img = cv2.imread('image.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float() / 255.0
    image_list = genNoise(img_tensor, 100)
    fig, axes = plt.subplots(5, 10, figsize=(100, 100))
    for idx, (ax, img) in enumerate(zip(axes.flat, image_list)):
        img_np = img.permute(1,0,2).cpu().numpy()
        img_np = img_np.clip(0, 1)
        ax.imshow(img_np)
        ax.axis('off')
    for ax in axes.flat[len(image_list):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()