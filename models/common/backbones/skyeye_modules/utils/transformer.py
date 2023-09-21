import torch
import numpy as np
import cv2


def computeM(scale, out_image_size, bev_focal_length, bev_camera_z):
    # Compute the mapping matrix from road to world (2D -> 3D)
    px_per_metre = abs((bev_focal_length * scale) / (bev_camera_z))
    shift = ((out_image_size[1] / 2 * scale) / 2, out_image_size[0] * 2 * scale)
    M = np.array([[1 / px_per_metre, 0, -shift[0] / px_per_metre],
                  [0, 1 / px_per_metre, -shift[1] / px_per_metre],
                  [0, 0, 0],  # This must be all zeros to cancel out the effect of Z
                  [0, 0, 1]], dtype=np.float)

    return M


def computeIntrinsicMatrix(fx, fy, px, py, img_scale):
    K = np.array([[fx * img_scale, 0, px * img_scale],
                  [0, fy * img_scale, py * img_scale],
                  [0, 0, 1]])
    return K


def computeExtrinsicMatrix(translation, rotation):
    # World to camera
    theta_w2c_x = np.deg2rad(rotation[0])
    theta_w2c_y = np.deg2rad(rotation[1])
    theta_w2c_z = np.deg2rad(rotation[2])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_w2c_x), -np.sin(theta_w2c_x)],
                    [0, np.sin(theta_w2c_x), np.cos(theta_w2c_x)]], dtype=np.float)
    R_y = np.array([[np.cos(theta_w2c_y), 0, np.sin(theta_w2c_y)],
                    [0, 1, 0],
                    [-np.sin(theta_w2c_y), 0, np.cos(theta_w2c_y)]], dtype=np.float)
    R_z = np.array([[np.cos(theta_w2c_z), -np.sin(theta_w2c_z), 0],
                    [np.sin(theta_w2c_z), np.cos(theta_w2c_z), 0],
                    [0, 0, 1]], dtype=np.float)

    R = (R_y @ (R_x @ R_z))

    t = -np.array(translation, dtype=np.float)
    t_rot = np.matmul(R, np.expand_dims(t, axis=1))

    extrinsic = np.zeros((3, 4), dtype=np.float)
    extrinsic[:3, :3] = R[:3, :3]
    extrinsic[:, 3] = t_rot.squeeze(1)

    return extrinsic


def computeHomography(intrinsic_matrix, extrinsic_matrix, M):
    P = np.matmul(intrinsic_matrix, extrinsic_matrix)
    H = np.linalg.inv(P.dot(M))

    return H

def getInitHomography(intrinsics, extrinsics, bev_params, img_scale, out_img_size):
    extrinsic_mat = computeExtrinsicMatrix(extrinsics['translation'], extrinsics['rotation'])
    intrinsic_mat = computeIntrinsicMatrix(intrinsics[0][0].item(), intrinsics[1][1].item(), intrinsics[0][2].item(), intrinsics[1][2].item(), img_scale)
    M = computeM(img_scale, out_img_size, bev_params['f'], bev_params['cam_z'])

    # print("Extrinsic", extrinsic_mat)
    # print("Intrinsic", intrinsic_mat)
    # print("M", M)

    H = computeHomography(intrinsic_mat, extrinsic_mat, M)

    H = torch.tensor(H.astype(np.float32))
    return H


if __name__ == "__main__":
    intrinsics = torch.tensor([[552.554261, 0, 682.049453],
                  [0, 552.554261, 238.769549],
                  [0, 0, 1]], dtype=torch.float)
    # intrinsics = {"fx": 552.554, "fy": 552.554, "px": 682.049453, "py": 238.769549}
    extrinsics = {"translation": (0.8, 0.3, 1.55), "rotation": (-85, 0, 90)}
    bev_params = {"f": 336, "cam_z": -24.1}
    scale = 1/2

    H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (384, 1408))
    H = H.numpy()

    img = cv2.imread("/home/gosalan/data/kitti360_dataset/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000200.png", cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
    warp_img = cv2.warpPerspective(img, H, (768, 704))

    import matplotlib.pyplot as plt
    plt.imshow(warp_img)
    plt.show()
    # cv2.imshow("Original", img)
    # cv2.imshow("Warped", warp_img)
    # cv2.waitKey(-1)

