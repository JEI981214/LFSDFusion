import numpy as np
import cv2
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
from tqdm import tqdm
import lpips
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import torch
import torchvision
from sewar.full_ref import uqi
from scipy.ndimage import uniform_filter
from sewar.utils import imresize
from skimage import img_as_ubyte


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -sum(h * np.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(),
                                                                                                   image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2


    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

#-----------------------------
    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)


        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        data_range = 255
        return ssim(image_F,image_A,data_range=data_range)+ssim(image_F,image_B,data_range=data_range)
    @classmethod
    def VIFF(cls,image_F, image_A, image_B):
        refA=image_A
        refB=image_B
        dist=image_F

        sigma_nsq = 2
        eps = 1e-10
        numA = 0.0
        denA = 0.0
        numB = 0.0
        denB = 0.0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
                refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                refA = refA[::2, ::2]
                refB = refB[::2, ::2]
                dist = dist[::2, ::2]

            mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
            mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq_A = mu1A * mu1A
            mu1_sq_B = mu1B * mu1B
            mu2_sq = mu2 * mu2
            mu1A_mu2 = mu1A * mu2
            mu1B_mu2 = mu1B * mu2
            sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
            sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
            sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

            sigma1A_sq[sigma1A_sq < 0] = 0
            sigma1B_sq[sigma1B_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            gA = sigma12_A / (sigma1A_sq + eps)
            gB = sigma12_B / (sigma1B_sq + eps)
            sv_sq_A = sigma2_sq - gA * sigma12_A
            sv_sq_B = sigma2_sq - gB * sigma12_B

            gA[sigma1A_sq < eps] = 0
            gB[sigma1B_sq < eps] = 0
            sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
            sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
            sigma1A_sq[sigma1A_sq < eps] = 0
            sigma1B_sq[sigma1B_sq < eps] = 0

            gA[sigma2_sq < eps] = 0
            gB[sigma2_sq < eps] = 0
            sv_sq_A[sigma2_sq < eps] = 0
            sv_sq_B[sigma2_sq < eps] = 0

            sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
            sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
            gA[gA < 0] = 0
            gB[gB < 0] = 0
            sv_sq_A[sv_sq_A <= eps] = eps
            sv_sq_B[sv_sq_B <= eps] = eps

            numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
            numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
            denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
            denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

        vifpA = numA / denA
        vifpB =numB / denB

        if np.isnan(vifpA):
            vifpA=1
        if np.isnan(vifpB):
            vifpB = 1
        return vifpA+vifpB

# LPIPS
class LPIPS_Calculator:
    def __init__(self, size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.transform = Compose([
            Resize((size, size)),
            ToTensor()
        ])
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.device = device

    def load_and_transform_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def calculate_lpips(self, folder_path1, folder_path2):
        lpips_values = []


        image_files1 = sorted(os.listdir(folder_path1))
        image_files2 = sorted(os.listdir(folder_path2))
        assert len(image_files1) == len(image_files2), "Mismatch in number of files in the folders."

        for file1, file2 in zip(image_files1, image_files2):
            image1 = self.load_and_transform_image(os.path.join(folder_path1, file1))
            image2 = self.load_and_transform_image(os.path.join(folder_path2, file2))

            image1 = image1.unsqueeze(0).to(self.device)
            image2 = image2.unsqueeze(0).to(self.device)

            lpips_value = self.lpips_model.forward(image1, image2)
            lpips_values.append(lpips_value.item())

        return (sum(lpips_values) / len(lpips_values)), lpips_values

    def calculate_IVF_lpips(self, folder_A, folder_B, folder_SR):
        A_average_lpips, A_lpips_values = self.calculate_lpips(folder_A, folder_SR)
        B_average_lpips, B_lpips_values = self.calculate_lpips(folder_B, folder_SR)

        return (A_average_lpips + B_average_lpips)/2, A_lpips_values, B_lpips_values


class MAE_Calculator:
    def __init__(self):
        pass

    def load_image_as_numpy_array(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def calculate_mae(self, folder_path1, folder_path2):
        mae_values = []


        image_files1 = sorted(os.listdir(folder_path1))
        image_files2 = sorted(os.listdir(folder_path2))
        # assert len(image_files1) == len(image_files2), (print(folder_path1), print(folder_path2),"Mismatch in number of files in the folders.")
        try:
            assert len(image_files1) == len(image_files2), "Mismatch in number of files in the folders."
        except AssertionError:
            print(folder_path1)
            print(folder_path2)
            raise
        for file1, file2 in zip(image_files1, image_files2):
            image1 = self.load_image_as_numpy_array(os.path.join(folder_path1, file1))
            image2 = self.load_image_as_numpy_array(os.path.join(folder_path2, file2))


            mae = np.mean(np.abs(image1.astype('float32') - image2.astype('float32')))
            mae_values.append(mae)

        return mae_values

    def calculate_IVF_mae(self, folder_A, folder_B, folder_SR):
        A_average_mae = self.calculate_mae(folder_A, folder_SR)
        B_average_mae = self.calculate_mae(folder_B, folder_SR)

        return  A_average_mae, B_average_mae

class RMSE_Calculator:
    def __init__(self):
        pass

    def load_image_as_numpy_array(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def calculate_rmse(self, folder_path1, folder_path2):
        rmse_values = []


        image_files1 = sorted(os.listdir(folder_path1))
        image_files2 = sorted(os.listdir(folder_path2))
        assert len(image_files1) == len(image_files2), "Mismatch in number of files in the folders."

        for file1, file2 in zip(image_files1, image_files2):
            image1 = self.load_image_as_numpy_array(os.path.join(folder_path1, file1))
            image2 = self.load_image_as_numpy_array(os.path.join(folder_path2, file2))


            rmse = np.sqrt(np.mean((image1.astype('float32') - image2.astype('float32')) ** 2))
            rmse_values.append(rmse)

        return rmse_values

    def calculate_IVF_rmse(self, folder_A, folder_B, folder_SR):
        A_average_rmse = self.calculate_rmse(folder_A, folder_SR)
        B_average_rmse = self.calculate_rmse(folder_B, folder_SR)

        return  A_average_rmse, B_average_rmse

# inception_model = torchvision.models.inception_v3(pretrained=True)


############################################################################

class QNRCalculator:
    def __init__(self, alpha=1, beta=1, p=1, q=1, r=4, ws=7):
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r
        self.ws = ws

    def load_image_as_numpy_array(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return np.array(image)

    def compute_qnr(self, pan, ms, fused):
        spectral_distortion = self.d_lambda(ms, fused, self.p)
        spatial_distortion = self.d_s(pan, ms, fused, q=self.q, ws=self.ws, r=self.r)
        return (1 - spectral_distortion) ** self.alpha * (1 - spatial_distortion) ** self.beta

    def calculate_qnr_for_folders(self, pan_folder, ms_folder, fused_folder):
        pan_files = sorted(os.listdir(pan_folder))
        ms_files = sorted(os.listdir(ms_folder))
        fused_files = sorted(os.listdir(fused_folder))

        assert len(pan_files) == len(ms_files) == len(fused_files), "Mismatch in number of files among folders."

        qnr_values = []

        for pan_file, ms_file, fused_file in zip(pan_files, ms_files, fused_files):
            pan_path = os.path.join(pan_folder, pan_file)
            ms_path = os.path.join(ms_folder, ms_file)
            fused_path = os.path.join(fused_folder, fused_file)

            pan = self.load_image_as_numpy_array(pan_path)
            ms = self.load_image_as_numpy_array(ms_path)
            fused = self.load_image_as_numpy_array(fused_path)


            qnr = self.compute_qnr(pan, ms, fused)
            qnr_values.append(qnr)

        return qnr_values

    def d_lambda(self, ms, fused, p=1):
        """calculates Spectral Distortion Index (D_lambda).

        :param ms: low resolution multispectral image.
        :param fused: high resolution fused image.
        :param p: parameter to emphasize large spectral differences (default = 1).

        :returns:  float -- D_lambda.
        """
        L = ms.shape[2]

        M1 = np.ones((L, L))
        M2 = np.ones((L, L))

        for l in range(L):
            for r in range(l + 1, L):
                M1[l, r] = M1[r, l] = uqi(fused[:, :, l], fused[:, :, r])
                M2[l, r] = M2[r, l] = uqi(ms[:, :, l], ms[:, :, r])

        diff = np.abs(M1 - M2) ** p
        return (1. / (L * (L - 1)) * np.sum(diff)) ** (1. / p)
    def d_s(self, pan, ms, fused, q=1, r=4, ws=7):
        """calculates Spatial Distortion Index (D_S)."""
        pan = pan.astype(np.float64)
        fused = fused.astype(np.float64)


        if len(pan.shape) == 3:
            pan = np.mean(pan, axis=2)

        # Apply uniform filter
        pan_degraded = uniform_filter(pan, size=ws) / (ws**2)

        # Normalize pan_degraded to [0, 1]
        pan_degraded = (pan_degraded - np.min(pan_degraded)) / (np.max(pan_degraded) - np.min(pan_degraded))

        # Convert to uint8
        pan_degraded = img_as_ubyte(pan_degraded)

        # Resize the image
        pan_degraded = imresize(pan_degraded, (pan.shape[0] // r, pan.shape[1] // r))  # (height, width)

        L = ms.shape[2]
        M1 = np.zeros(L)
        M2 = np.zeros(L)

        for l in range(L):
            # Resize ms[:, :, l] to match pan_degraded's shape
            ms_resized = imresize(ms[:, :, l], (pan_degraded.shape[0], pan_degraded.shape[1]))  # (height, width)

            M1[l] = uqi(fused[:, :, l], pan)
            target_size = (pan.shape[0] // r, pan.shape[1] // r)  # (height, width)
            #print(ms_resized.shape, pan_degraded.shape)
            pan_degraded = imresize(pan_degraded, target_size)
            ms_resized = imresize(ms[:, :, l], target_size)
            #print(ms_resized.shape, pan_degraded.shape)

            M2[l] = uqi(ms_resized, pan_degraded)

        diff = np.abs(M1 - M2) ** q
        return ((1. / L) * (np.sum(diff))) ** (1. / q)


def calculate_metrics(ori_A_folder, ori_B_folder, fusion_folder, save_path):
    results = []
    a_map = {os.path.splitext(f)[0]: f for f in os.listdir(ori_A_folder)}
    b_map = {os.path.splitext(f)[0]: f for f in os.listdir(ori_B_folder)}


    fusion_files = sorted(f for f in os.listdir(fusion_folder)
                          if os.path.splitext(f)[-1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    for img_name in tqdm(fusion_files, desc="Processing Images"):
        stem = os.path.splitext(img_name)[0]
        A = image_read_cv2(os.path.join(ori_A_folder, a_map[stem]), 'GRAY')
        B = image_read_cv2(os.path.join(ori_B_folder, b_map[stem]), 'GRAY')

        fi = image_read_cv2(os.path.join(fusion_folder, img_name), 'GRAY')
        if fi.shape != A.shape:
            fi = cv2.resize(fi, (A.shape[1], A.shape[0]))


        print(f"Processing {img_name}:")
        print(f"A shape: {A.shape}, B shape: {B.shape}, fi shape: {fi.shape}")


        metrics = [Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi), Evaluator.AG(fi),
                   Evaluator.MI(fi, A, B), Evaluator.MSE(fi, A, B), Evaluator.VIFF(fi, A, B),
                   Evaluator.Qabf(fi, A, B), Evaluator.SSIM(fi, A, B), Evaluator.PSNR(fi, A, B),
                   Evaluator.SCD(fi, A, B), Evaluator.CC(fi, A, B)]
        results.append(metrics)


    lpips_calculator = LPIPS_Calculator()
    lpips_value, A_lpips_values, B_lpips_values = lpips_calculator.calculate_IVF_lpips(ori_A_folder, ori_B_folder, fusion_folder)
    mae_calculator = MAE_Calculator()
    A_mae, B_mae = mae_calculator.calculate_IVF_mae(ori_A_folder, ori_B_folder, fusion_folder)
    rmse_calculator = RMSE_Calculator()
    A_rmse, B_rmse = rmse_calculator.calculate_IVF_rmse(ori_A_folder, ori_B_folder, fusion_folder)
    qnr_calculator = QNRCalculator()
    qnr_scores = qnr_calculator.calculate_qnr_for_folders(ori_A_folder, ori_B_folder, fusion_folder)


    for i in range(len(results)):
        avg_lpips = (A_lpips_values[i] + B_lpips_values[i]) / 2
        avg_mae = (A_mae[i] + B_mae[i]) / 2
        avg_rmse = (A_rmse[i] + B_rmse[i] ) / 2
        qnr = qnr_scores[i]


        results[i].extend([avg_lpips, avg_mae, avg_rmse,qnr])


    columns = ['EN', 'SD', 'SF', 'AG', 'MI', 'MSE', 'VIF', 'Qabf', 'SSIM', 'PSNR', 'SCD', 'CC', 'LPIPS', 'MAE', 'RMSE','qnr']
    df = pd.DataFrame(results, columns=columns)


    averages_list = np.mean(df.values, axis=0).tolist()


    average_row = averages_list
    average_df = pd.DataFrame([average_row], columns=df.columns)


    average_df.insert(0, 'Label', 'Average')
    df.insert(0, 'Label', [f'Image_{i+1}' for i in range(len(results))])


    if len(average_row) == len(df.columns) - 1:

        df = pd.concat([df, average_df], ignore_index=True)
    else:
        print(f"Mismatch: average_row length = {len(average_row)}, df.columns length = {len(df.columns)}")


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_excel(save_path, index=False)

    print(f"Processing Complete. Results saved to {save_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LFSDFusion fusion evaluation metrics (16 metrics -> xlsx)")
    ap.add_argument("--ref-a", required=True, help="reference modality A (clear source) folder")
    ap.add_argument("--ref-b", required=True, help="reference modality B (clear source) folder")
    ap.add_argument("--fusion", required=True, help="fused images folder")
    ap.add_argument("--out", required=True, help="output .xlsx path")
    args = ap.parse_args()
    calculate_metrics(args.ref_a, args.ref_b, args.fusion, args.out)
