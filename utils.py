import numpy as np

def calculate_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, L=255, K1=0.01, K2=0.03, window_size=11):
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    def gaussian_window(window_size, sigma=1.5):
        gauss = np.outer(
            np.hanning(window_size),
            np.hanning(window_size)
        )
        return gauss / np.sum(gauss)

    window = gaussian_window(window_size)
    
    mu1 = apply_filter(img1, window)
    mu2 = apply_filter(img2, window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = apply_filter(img1 * img1, window) - mu1_sq
    sigma2_sq = apply_filter(img2 * img2, window) - mu2_sq
    sigma12 = apply_filter(img1 * img2, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def apply_filter(img, window):
    filtered_img = np.zeros_like(img, dtype=float)
    pad_size = window.shape[0] // 2

    if img.ndim == 3:  # For RGB images
        for i in range(3):
            filtered_img[..., i] = convolve2d(img[..., i], window, pad_size)
    else:  # For grayscale images
        filtered_img = convolve2d(img, window, pad_size)

    return filtered_img

def convolve2d(img, window, pad_size):
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
    filtered_img = np.zeros_like(img, dtype=float)
    
    for i in range(filtered_img.shape[0]):
        for j in range(filtered_img.shape[1]):
            filtered_img[i, j] = np.sum(padded_img[i:i + window.shape[0], j:j + window.shape[1]] * window)
    
    return filtered_img
