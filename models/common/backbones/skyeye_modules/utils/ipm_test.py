import kornia
import cv2
import numpy as np
import torch
import torch.nn.functional as F

theta_init = {"front": np.array([[-1.75686906e-17, -8.29247445e-01,  1.44420026e+02],
                                 [-2.43234031e-02, -4.51341365e-01,  1.02537488e+02],
                                 [-2.93556429e-20, -1.16315501e-03,  2.21496486e-01]]),
              "left": np.array([[-2.01149112e-02, -5.39502057e-01, 1.90716522e+02],
                                [-2.46336616e-18, -3.03595165e-01, 1.19698308e+02],
                                [-1.36961767e-35, -7.67510347e-04, 2.68628622e-01]]),
              "right": np.array([[1.80956618e-02, -4.85343769e-01, 1.68169327e+02],
                                 [-0.00000000e+00, -2.57157252e-01, 7.65650381e+01],
                                 [0.00000000e+00, -6.90463288e-04, 2.41662151e-01]])}

# H_dim = (384, 1408)
# calib = cv2.resize(cv2.imread("/home/gosalan/Documents/po_bev/po_bev_utils/other/calib.png", cv2.IMREAD_UNCHANGED), dsize=(0, 0), fx=1, fy=1).astype(np.float32)
# calib = calib[:H_dim[0], :H_dim[1], :]
# calib_dim = (calib.shape[0], calib.shape[1])

def generateMeshGrid(height, width, device):
    # Generate Coordinates
    xs = torch.linspace(0, width-1, width, device=device, dtype=torch.float)
    ys = torch.linspace(0, height-1, height, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid = torch.stack(torch.meshgrid([ys, xs], indexing='ij'))  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2

def transformPoints(points, transform):
    B, H, W, C = points.shape

    points = points.reshape(-1, points.shape[1] * points.shape[2], 2).flip(dims=[-1]).transpose(1, 2)
    ones = torch.ones((points.shape[0], 1, points.shape[2]))
    points = torch.cat((points, ones), dim=1)

    # ones = torch.ones((points.shape[0], points.shape[1], points.shape[2], 1))
    # points = torch.cat((points, ones), dim=3)

    points_warp = torch.matmul(transform, points)
    points_warp = (points_warp / points_warp[:, 2, :].unsqueeze(1))[:, :2, :]
    points_warp = points_warp.transpose(1, 2).flip(dims=[-1]).reshape(B, H, W, C)

    return points_warp

def getNormalised(grid_warp):
    half_height = grid_warp.shape[1]
    half_width = grid_warp.shape[2]
    grid_warp[:, :, :, 1] = 2 * (grid_warp[:, :, :, 1] / half_width) - 1
    grid_warp[:, :, :, 0] = 2 * (grid_warp[:, :, :, 0] / half_height) - 1

    return grid_warp


def adjustHomography(in_homography, in_dsize, out_dsize):
    fx = out_dsize[1] / in_dsize[1]
    fy = out_dsize[0] / in_dsize[0]
    Ko = np.array([[fx, 0, 0],
                   [0, fy, 0],
                   [0, 0, 1]], dtype=np.float32)

    return Ko.dot(in_homography)


def computeM(scale, image_size, bev_focal_length, bev_camera_z):
    # Compute the mapping matrix from road to world (2D -> 3D)
    px_per_metre = abs((bev_focal_length * scale) / (bev_camera_z - 0.9))
    # shift = ((image_size[1] * scale) / 2, (image_size[0] * scale) / 2)
    # shift = ((image_size[1] / 2 * scale) / 2, image_size[0] * 2 * scale)
    shift = ((image_size[1] / 2 * scale), image_size[0] * scale * 2)
    M = np.array([[1 / px_per_metre, 0, -shift[0] / px_per_metre],
                  [0, 1 / px_per_metre, -shift[1] / px_per_metre],
                  [0, 0, 0],  # This must be all zeros to cancel out the effect of Z
                  [0, 0, 1]])

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

def getInitHomography(intrinsics, extrinsics, bev_params, img_scale, img_size):
    extrinsic_mat = computeExtrinsicMatrix(extrinsics['translation'], extrinsics['rotation'])
    intrinsic_mat = computeIntrinsicMatrix(intrinsics['fx'], intrinsics['fy'], intrinsics['px'], intrinsics['py'], img_scale)
    M = computeM(img_scale, img_size, bev_params['f'], bev_params['cam_z'])

    H = computeHomography(intrinsic_mat, extrinsic_mat, M)

    H = torch.tensor(H.astype(np.float32))
    return H


if __name__ == "__main__":
    # Checkerboard
    # intrinsics = {"fx": 552.554, "fy": 552.554, "px": 487, "py": 343}
    # extrinsics = {"translation": (0.8, 0.3, 1.55), "rotation": (-90, 0, 90)}
    # bev_params = {"f": 336, "cam_z": -24.1}
    # scale = 1
    # H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (686, 975))

     # KITTI
    intrinsics = {"fx": 552.554, "fy": 552.554, "px": 682.049453, "py": 238.769549}
    extrinsics = {"translation": (0.8, 0.3, 1.55), "rotation": (-90, 0, 180)}
    bev_params = {"f": 336, "cam_z": -24.1}
    scale = 1
    H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (376, 1408))

    # nuScenes
    # intrinsics = {"fx": 1266.4172, "fy": 1266.4172, "px": 816.267, "py": 491.507}
    # extrinsics = {"translation": (0., 0.6, 1.85), "rotation": (-90, 0, 180)}
    # bev_params = {"f": 336, "cam_z": 26}
    # scale = 0.5
    # H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (900, 1600))

    # BEV Image
    # intrinsics = {"fx": 552.554, "fy": 552.554, "px": 800, "py": 476}
    # extrinsics = {"translation": (0.8, 0.3, 1.55), "rotation": (-90, 0, 90)}
    # bev_params = {"f": 336, "cam_z": -24.1}
    # scale = 1
    # H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (952, 1600))

    # Cam2Bev
    # intrinsics = {"fx": 278.283, "fy": 408.1295, "px": 256, "py": 128}
    # extrinsics = {"translation": (0.6, 0, 1.4), "rotation": (-90, 0, 90)}
    # bev_params = {"f": 682.578, "cam_z": -49.1}
    # scale = 1
    # H = getInitHomography(intrinsics, extrinsics, bev_params, scale, (256, 512))

    H = H.numpy()
    # H = np.array([[-2.187729553235505e-18, 0.23731675561238333, -43.22401470771198], [0.024323414544098858, 0.8261133899392912, -213.84049312356058], [-4.6756388370332186e-21, 0.0011675986244287057, -0.27878699696786247]])

    # Converting from Cam2BEV matrix to OpenCV matrix
    # Front
    # H = np.array([[4.651574574230558e-14, 10.192351107009959, -5.36318723862984e-07], [-5.588661045867985e-07, 0.0, 2.3708767903941617], [35.30731833118676, 0.0, -1.7000018578614013]])
    # Rear
    # H = np.array([[-5.336674306912119e-14, -10.192351107009957, 5.363187220578325e-07], [5.588660952931949e-07, 3.582264351370481e-23, 2.370876772982613], [-35.30731833118661, -2.263156574813233e-15, -0.5999981421386035]])  # rear
    # Left
    # H = np.array([[20.38470221401992, 7.562206982469407e-14, -0.28867638384075833], [-3.422067857504854e-23, 2.794330463189411e-07, 2.540225111648729], [2.1619497190382224e-15, -17.65365916559334, -0.4999990710692976]])
    # Right
    # H = np.array([[-20.38470221401991, -4.849709834037436e-15, 0.2886763838407495], [-3.4220679184765114e-23, -2.794330512976549e-07, 2.5402251116487626], [2.161949719038217e-15, 17.653659165593304, -0.5000009289306967]])
    #
    # fx = intrinsics['px']
    # fy = intrinsics['py']
    # px = intrinsics['px']
    # py = intrinsics['py']
    # Si = np.array([[fx, 0, px],
    #                [0, fy, py],
    #                [0, 0, 1]], dtype=np.float32)
    #
    # # scale from output resolution back to unit grid (-1,1)^2
    # fx = 1 / intrinsics['px']
    # fy = 1 / intrinsics['py']
    # px = -1
    # py = -1
    # So = np.array([[fx, 0, px],
    #                [0, fy, py],
    #                [0, 0, 1]], dtype=np.float32)
    # H = np.linalg.inv(Si @ H @ So)

    # img = cv2.imread("/home/gosalan/Documents/po_bev/po_bev_utils/homography/t_0_0_0046000.png", cv2.IMREAD_UNCHANGED)
    # h, w = img.shape[0]//2, img.shape[1]//2
    # img = img[h-128:h+128, w-256:w+256, :]

    # img = cv2.imread("/home/gosalan/Documents/po_bev/po_bev_utils/homography/calib.png", cv2.IMREAD_UNCHANGED)
    img = cv2.imread("/home/gosalan/data/kitti360_dataset/data_2d_raw/2013_05_28_drive_0002_sync/image_00/data_rect/0000005430.png", cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(
    #     "/home/gosalan/Documents/data/nuScenes_dataset/nuScenes/samples/CAM_FRONT/n015-2018-09-25-11-10-38+0800__CAM_FRONT__1537845441862460.jpg",
    #     cv2.IMREAD_UNCHANGED)
    print(img.shape)
    img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
    # img = cv2.rotate(cv2.imread("/home/gosalan/Documents/data/kitti360_dataset/data_2d_semantics_bev_affine/2013_05_28_drive_0003_sync/panoptic_rgb_filled/0000000275.png", cv2.IMREAD_UNCHANGED),
    #                  cv2.ROTATE_180)
    # img = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
    print(H)
    print(img.shape[1] // 2, img.shape[0] * 2)
    # warp_img = cv2.warpPerspective(img, H, (img.shape[1] // 2, img.shape[0] * 2))
    # warp_img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]*2))

    import kornia
    img_torch = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).type(torch.float)
    H_torch = torch.tensor(H).view(-1, 3, 3).type(torch.float)
    warp_img_torch = kornia.geometry.transform.warp_perspective(img_torch, H_torch, (img.shape[0] * 2, img.shape[1]))

    reconst_img_torch = kornia.geometry.transform.warp_perspective(warp_img_torch, torch.inverse(H_torch), (img.shape[0], img.shape[1]))

    warp_img_torch = torch.rot90(warp_img_torch, k=2, dims=[2, 3])  #.squeeze(0).type(torch.uint8)
    warp_img_torch = torch.flip(warp_img_torch, dims=[3]).squeeze(0).type(torch.uint8)
    # warp_img_torch = torch.rot90(warp_img_torch, k=1, dims=[2, 3]).squeeze(0).type(torch.uint8)
    warp_img = warp_img_torch.permute(1, 2, 0).cpu().numpy()
    reconst_img_torch = reconst_img_torch.squeeze(0).type(torch.uint8).permute(1, 2, 0).cpu().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(warp_img)
    plt.show()
    #
    # cv2.imshow("Original", img)
    # cv2.imshow("Warped", warp_img)
    # cv2.imshow("Reconstruct", reconst_img_torch)
    # cv2.waitKey(-1)




# for key in theta_init.keys():
#     if key == "front":
#         in_dsize = (calib.shape[0], calib.shape[1])
#         out_dsize = (calib.shape[0], calib.shape[1])
#
#         image = torch.from_numpy(calib)[:, :, :3].permute(2, 0, 1).unsqueeze(0)
# #######################################################################################################################
#         # WARP SOLUTION 1
#         # Adjust the homography to match the output resolution
#         # stn_theta = np.linalg.inv(theta_init[key]).astype(np.float32)
#         # stn_theta = adjustHomography(stn_theta, in_dsize, out_dsize)
#         #
#         # stn_theta = torch.from_numpy(stn_theta).view(-1, 3, 3)
#         #
#         # grid = generateMeshGrid(out_dsize[0], out_dsize[1], image.device)
#         # grid_warp = transformPoints(grid, stn_theta.type(grid.dtype))
#         # grid_warp = getNormalised(grid_warp)
#         # grid_warp = torch.flip(grid_warp, dims=[3])
#         # calib_warp = F.grid_sample(image, grid_warp, align_corners=True)
# #######################################################################################################################
#         # WARP SOLUTION 2
#         stn_theta = torch.from_numpy(theta_init[key]).view(-1, 3, 3).type(torch.FloatTensor)
#         calib_warp = kornia.warp_perspective(image, stn_theta, out_dsize)
# #######################################################################################################################
#
#         # Visualisation and OpenCV
#         calib_warp = calib_warp.squeeze(0).permute(1, 2, 0).numpy()
#         calib = image.squeeze(0).permute(1, 2, 0).numpy()
#
#         ipm_image = cv2.warpPerspective(calib, theta_init[key], (out_dsize[1], out_dsize[0]), flags=cv2.INTER_LINEAR)
#         cv2.imshow("Original", calib.astype(np.uint8))
#         cv2.imshow("Warp", calib_warp.astype(np.uint8))
#         cv2.imshow("IPM", ipm_image.astype(np.uint8))
#         cv2.waitKey(-1)
