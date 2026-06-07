import cv2
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# Parameters
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
BLOCK_SIZE = 16  # for orientation

def preprocess_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    eq = cv2.equalizeHist(gray)

    # Median blur
    blur = cv2.medianBlur(eq, 3)

    # Gabor filter (ridge enhancement)
    kernel = cv2.getGaborKernel((21, 21), 3, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
    enhanced = cv2.filter2D(blur, cv2.CV_8UC3, kernel)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

def extract_features(img, filename):
    features = {}

    # --- 1. Ridge density (naive count of black pixels / total)
    black_pixels = np.sum(img == 0)
    total_pixels = img.size
    features["ridge_density"] = black_pixels / total_pixels

    # --- 2. Minutiae count (simplified: count ridge endings from skeleton)
    skeleton = cv2.ximgproc.thinning(img)
    minutiae_count = np.sum(skeleton == 255) // 50  # rough proxy
    features["minutiae_count"] = minutiae_count

    # --- 3. Pattern type (very naive: based on symmetry)
    h, w = img.shape
    left = np.sum(img[:, :w//2] == 0)
    right = np.sum(img[:, w//2:] == 0)
    if abs(left - right) < 0.05 * total_pixels:
        pattern = "whorl"
    elif left > right:
        pattern = "loop-left"
    else:
        pattern = "loop-right"
    features["pattern_type"] = pattern

    # --- 4. Orientation variance (block-wise gradient angles)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)
    orientation_variance = np.var(angles)
    features["orientation_variance"] = orientation_variance

    # --- 5. Ridge frequency (distance between ridges using FFT)
    freq = np.mean(np.fft.fftshift(np.abs(np.fft.fft2(img))))
    features["ridge_frequency"] = freq

    # --- 6. Local Binary Pattern histogram
    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, LBP_POINTS + 3),
                             range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= hist.sum()  # normalize
    for i, val in enumerate(hist[:10]):  # take first 10 bins
        features[f"lbp_{i}"] = val

    # --- 7. Entropy (extra texture measure)
    features["entropy"] = shannon_entropy(img)

    features["filename"] = filename
    return features

def process_fingerprint_folder(folder, output_csv="fingerprint_features.csv"):
    rows = []
    for file in os.listdir(folder):
        if file.endswith((".png", ".jpg", ".bmp", ".jpeg")):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            preprocessed = preprocess_image(img)
            feats = extract_features(preprocessed, file)
            rows.append(feats)

    # Save to CSV
    keys = rows[0].keys()
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Features saved to {output_csv}")
