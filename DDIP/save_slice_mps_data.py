import os
import numpy as np
import scipy.io as sio

# =========================
# 경로 설정
# =========================
mat_path = "/home/lee-ho-hyeon/바탕화면/단일영상재건MoDL/MRI_tutorial_data.mat"

save_root = "/home/lee-ho-hyeon/바탕화면/DDIP/data/tutorial_brain/vol001"

slice_dir = os.path.join(save_root, "slice")
mps_dir = os.path.join(save_root, "mps")

os.makedirs(slice_dir, exist_ok=True)
os.makedirs(mps_dir, exist_ok=True)

# =========================
# 320x320 zero padding 함수
# =========================
def pad_to_320(img):

    H, W = img.shape

    target_H = 320
    target_W = 320

    out = np.zeros((target_H, target_W), dtype=img.dtype)

    start_h = (target_H - H) // 2
    start_w = (target_W - W) // 2

    out[
        start_h:start_h + H,
        start_w:start_w + W
    ] = img

    return out


# =========================
# 데이터 로드
# =========================
mat = sio.loadmat(mat_path)

kData = mat["kData"]            # (206,176,32)
coil_sens = mat["coil_sens"]    # (206,176,32)

print("kData:", kData.shape)
print("coil_sens:", coil_sens.shape)

# =========================
# Coil Image 생성
# =========================
coil_imgs = np.fft.ifftshift(
    np.fft.ifft2(
        np.fft.fftshift(kData, axes=(0, 1)),
        axes=(0, 1)
    ),
    axes=(0, 1)
)

# =========================
# SENSE Coil Combination
# =========================
denom = np.sum(
    np.abs(coil_sens) ** 2,
    axis=2
) + 1e-8

img = np.sum(
    np.conj(coil_sens) * coil_imgs,
    axis=2
) / denom
img = img / np.abs(img).max()

# =========================
# slice padding
# =========================
img_pad = pad_to_320(img)

# =========================
# mps 생성
# (coil,H,W)
# =========================
mps = np.transpose(
    coil_sens,
    (2, 0, 1)
)

# coil별 padding
mps_pad = np.zeros(
    (mps.shape[0], 320, 320),
    dtype=np.complex64
)

for c in range(mps.shape[0]):
    mps_pad[c] = pad_to_320(mps[c])

# =========================
# 저장
# =========================
slice_path = os.path.join(slice_dir, "000.npy")
mps_path = os.path.join(mps_dir, "000.npy")

np.save(slice_path, img_pad.astype(np.complex64))
np.save(mps_path, mps_pad.astype(np.complex64))

print()
print("Saved:")
print(slice_path)
print(mps_path)

print()
print("slice shape :", img_pad.shape)
print("mps shape   :", mps_pad.shape)