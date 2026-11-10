import cv2
import numpy as np


POISSON_PEAKS = {"light": 160, "medium": 60, "heavy": 25}
RICIAN_SIGMAS = {"light": 8, "medium": 20, "heavy": 40}
BLUR_ANGLES = [0, 45, 90]
BLUR_KERNEL_SIZE = 5
BLUR_ANGLE_JITTER = 5.0


def _rng(seed):
    return np.random.default_rng(int(seed))


def motion_blur(img, angle_deg, rng, kernel_size=BLUR_KERNEL_SIZE):

    if angle_deg not in BLUR_ANGLES:
        raise ValueError(angle_deg)
    ang = angle_deg + float(rng.uniform(-BLUR_ANGLE_JITTER, BLUR_ANGLE_JITTER))
    k = int(kernel_size)
    kern = np.zeros((k, k), dtype=np.float32)
    center = k // 2
    rad = np.deg2rad(ang)
    for i in range(k):
        dx = i - center
        x = int(round(center + dx * np.cos(rad)))
        y = int(round(center + dx * np.sin(rad)))
        if 0 <= x < k and 0 <= y < k:
            kern[y, x] = 1.0
    kern /= max(kern.sum(), 1e-6)
    out = cv2.filter2D(img, -1, kern)
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_noise(img, sigma, rng):

    noise = rng.normal(0, float(sigma), img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def poisson_noise(img, peak, rng):

    img_f = img.astype(np.float32) / 255.0
    scaled = img_f * float(peak)
    noisy = rng.poisson(np.maximum(scaled, 1e-6)).astype(np.float32) / float(peak) * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rician_noise(img, sigma, rng):
    """Rician:sqrt((I+n1)² + n2²),n1,n2 ~ N(0,σ)."""
    img_f = img.astype(np.float32)
    n1 = rng.normal(0, float(sigma), img_f.shape).astype(np.float32)
    n2 = rng.normal(0, float(sigma), img_f.shape).astype(np.float32)
    out = np.sqrt((img_f + n1) ** 2 + n2 ** 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def stripe_noise(img, variance, rng):

    bias = rng.normal(0, np.sqrt(float(variance)), img.shape[1]).astype(np.float32)
    return np.clip(img.astype(np.float32) + bias[None, :, None], 0, 255).astype(np.uint8)


def bilinear_downsample(img, factor):

    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)


def rigid_transform(img, rng, theta_max=7.0, t_max=0.07):

    h, w = img.shape[:2]
    theta = float(rng.uniform(-theta_max, theta_max))
    tx = float(rng.uniform(-t_max, t_max)) * w
    ty = float(rng.uniform(-t_max, t_max)) * h
    m = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def nonrigid_transform(img, rng, smooth_sigma=15, n_smooth=6):

    h, w = img.shape[:2]
    dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    for _ in range(n_smooth):
        dx = cv2.GaussianBlur(dx, (0, 0), smooth_sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), smooth_sigma)
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def degrade_ir(img, seed, downsample=4, stripe_var=15,
               misreg="rigid", theta_max=7.0, t_max=0.07):

    rng = _rng(seed)
    out = bilinear_downsample(img, downsample)
    out = stripe_noise(out, stripe_var, rng)
    if misreg == "rigid":
        out = rigid_transform(out, rng, theta_max, t_max)
    elif misreg == "nonrigid":
        out = nonrigid_transform(out, rng)
    elif misreg is not None:
        raise ValueError(misreg)
    return out


def degrade_vi(img, seed, gaussian_var=15, blur_angle=None):

    rng = _rng(seed)
    if blur_angle is None:
        blur_angle = BLUR_ANGLES[int(rng.integers(0, len(BLUR_ANGLES)))]
    out = motion_blur(img, blur_angle, rng)
    return gaussian_noise(out, gaussian_var, rng)


def degrade_medical(img, modality, seed, level="light", blur_angle=None):

    rng = _rng(seed)
    if blur_angle is None:
        blur_angle = BLUR_ANGLES[int(rng.integers(0, len(BLUR_ANGLES)))]
    out = motion_blur(img, blur_angle, rng)
    if modality == "MRI":
        out = rician_noise(out, RICIAN_SIGMAS[level], rng)
    elif modality in ("CT", "PET", "SPECT"):
        out = poisson_noise(out, POISSON_PEAKS[level], rng)
    else:
        raise ValueError(modality)
    return out
