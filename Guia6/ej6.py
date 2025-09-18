import cv2 as cv
import numpy as np

def calcPSF(filterSize, R):
    h = np.zeros(filterSize, dtype=np.float32)
    center = (filterSize[1] // 2, filterSize[0] // 2)
    cv.circle(h, center, R, 255, -1, 8)
    h /= np.sum(h)
    return h

def filter2DFreq(inputImg, H):
    F = np.fft.fft2(inputImg)
    F_shifted = np.fft.fftshift(F)
    G = F_shifted * H
    img_back = np.fft.ifft2(np.fft.ifftshift(G))
    return np.abs(img_back)

def calcWnrFilter(input_h_PSF, nsr):
    H = np.fft.fft2(np.fft.fftshift(input_h_PSF))
    H_conj = np.conj(H)
    denom = (np.abs(H)**2 + nsr)
    output_G = H_conj / denom
    return output_G

def update(val):
    R = cv.getTrackbarPos('R', 'Deblurred')
    snr = cv.getTrackbarPos('SNR', 'Deblurred')
    snr = max(snr, 1)

    h = calcPSF(img_roi.shape, R)
    Hw = calcWnrFilter(h, 1.0 / snr)
    imgOut = filter2DFreq(img_roi, Hw)
    imgOut = cv.normalize(imgOut, None, 0, 255, cv.NORM_MINMAX)
    imgOut = np.uint8(imgOut)
    cv.imshow('Deblurred', imgOut)

def main():
    global img_roi
    image_path = r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\original.jpg'
    imgIn = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if imgIn is None:
        print("ERROR: Image cannot be loaded..!!")
        return

    roi = (slice(0, imgIn.shape[0] & -2), slice(0, imgIn.shape[1] & -2))
    img_roi = imgIn[roi]

    cv.namedWindow('Deblurred', cv.WINDOW_NORMAL)
    cv.createTrackbar('R', 'Deblurred', 1, 100, update)
    cv.createTrackbar('SNR', 'Deblurred', 1, 10000, update)

    update(None)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
