import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def ruido_sp(img, probs, probp, valors, valorp):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            if r < probp:
                img[i, j] = valorp
            elif r < probs + probp:
                img[i, j] = valors
    return img

def ruido_exp(img, a):
    img_ruido = img.copy().astype(float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            ruido = -np.log(r) / a
            img_ruido[i, j] += ruido
    img_ruido = np.clip(img_ruido, 0, 255).astype(np.uint8)
    return img_ruido
if __name__ == "__main__":
    fig, axs = plt.subplots(5, 2, figsize=(10, 16))
    plt.subplots_adjust(hspace=0.4)

    # Ruido Gaussiano
    img = np.ones((600, 600), dtype=np.uint8)
    ruido_gauss = cv.randn(img, 127, 20)
    histo_gauss = cv.calcHist([ruido_gauss], [0], None, [256], [0, 256])
    axs[0, 0].imshow(ruido_gauss, cmap='gray')
    axs[0, 0].set_title('Ruido Gaussiano')
    axs[0, 0].axis('off')
    axs[0, 1].bar(range(256), histo_gauss.ravel(), color='gray')

    # Ruido Uniforme
    img = np.ones((600, 600), dtype=np.uint8) * 127
    ruido_uniform = cv.randu(img, 127 - 15, 127 + 15)
    histo_uniform = cv.calcHist([ruido_uniform], [0], None, [256], [0, 256])
    axs[1, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    axs[1, 0].set_title('Ruido Uniforme')
    axs[1, 0].axis('off')
    axs[1, 1].bar(range(256), histo_uniform.ravel(), color='gray')

    # Ruido Sal y Pimienta
    img = np.ones((600, 600), dtype=np.uint8) * 127
    ruido_sp_img = ruido_sp(img, 0.1, 0.1, 235, 20)
    histo_sp = cv.calcHist([ruido_sp_img], [0], None, [256], [0, 256])
    axs[2, 0].imshow(ruido_sp_img, cmap='gray', vmin=0, vmax=255)
    axs[2, 0].set_title('Ruido Sal y Pimienta')
    axs[2, 0].axis('off')
    axs[2, 1].bar(range(256), histo_sp.ravel(), color='gray')

    # Ruido Impulsivo Unipolar
    img = np.ones((600, 600), dtype=np.uint8) * 127
    ruido_impulsivo = ruido_sp(img.copy(), 0.2, 0, 0, 0)  # solo valor bajo
    histo_impulsivo = cv.calcHist([ruido_impulsivo], [0], None, [256], [0, 256])
    axs[3, 0].imshow(ruido_impulsivo, cmap='gray', vmin=0, vmax=255)
    axs[3, 0].set_title('Ruido Impulsivo Unipolar')
    axs[3, 0].axis('off')
    axs[3, 1].bar(range(256), histo_impulsivo.ravel(), color='gray')

    # Ruido Exponencial
    img = np.ones((600, 600), dtype=np.uint8) * 127
    ruido_exp_img = ruido_exp(img, 0.1)
    histo_exp = cv.calcHist([ruido_exp_img], [0], None, [256], [0, 256])
    axs[4, 0].imshow(ruido_exp_img, cmap='gray', vmin=0, vmax=255)
    axs[4, 0].set_title('Ruido Exponencial')
    axs[4, 0].axis('off')
    axs[4, 1].bar(range(256), histo_exp.ravel(), color='gray')

    plt.tight_layout()
    plt.show()
