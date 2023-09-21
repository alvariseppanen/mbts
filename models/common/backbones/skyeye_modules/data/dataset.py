import glob
from itertools import chain
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import umsgpack
from PIL import Image
import json
from po_bev_unsupervised.data.transform import *


class BEVKitti360Dataset(data.Dataset):
    """Instance segmentation dataset

    This assumes the dataset to be formatted as defined in:
        https://github.com/mapillary/seamseg/wiki/Dataset-format

    Parameters
    ----------
    seam_root_dir : str
        Path to the root directory of the dataset
    split_name : str
        Name of the split to load: this must correspond to one of the files in `root_dir/lst`
    transform : callable
        Transformer function applied to the loaded entries to prepare them for pytorch. This should be callable as
        `transform(img, msk, cat, cls)`, where:
            - `img` is a PIL.Image with `mode="RGB"`, containing the RGB data
            - `msk` is a list of PIL.Image with `mode="L"`, containing the instance segmentation masks
            - `cat` is a list containing the instance id to class id mapping
            - `cls` is an integer specifying a requested class for class-uniform sampling, or None

    """
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _BEV_PLABEL_DIR = "bev_plabel_dynamic"
    _FV_MSK_DIR = "front_msk_seam"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _BEV_METADATA_FILE = "metadata_ortho.bin"
    _FV_METADATA_FILE = "metadata_front.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform, window=0, use_offline_plabels=False):
        super(BEVKitti360Dataset, self).__init__()
        self.seam_root_dir = seam_root_dir  # Directory of seamless data
        self.kitti_root_dir = dataset_root_dir  #  Directory of the KITTI360 data
        self.split_name = split_name
        self.transform = transform
        self.window = window  # Single-sided window count. The number of images samples is [i - window to i + window]
        self.rgb_cameras = ['front']
        self.use_offline_plabels = use_offline_plabels

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_MSK_DIR, BEVKitti360Dataset._BEV_DIR)
        self._bev_plabel_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_PLABEL_DIR, BEVKitti360Dataset._BEV_DIR)
        self._fv_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._FV_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)

        # Load meta-data and split
        self._bev_meta, self._bev_images, self._bev_images_all, self._fv_meta, self._fv_images, self._fv_images_all,\
        self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._BEV_METADATA_FILE), "rb") as fid:
            bev_metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._FV_METADATA_FILE), 'rb') as fid:
            fv_metadata = umsgpack.unpack(fid, encoding="utf-8")

        # Read the files for this split
        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        # Get all the frames in the train dataset. This will be used for generating samples for temporal consistency.
        if self.split_name == "train":
            with open(os.path.join(self._lst_dir, "{}_all.txt".format(self.split_name)), 'r') as fid:
                lst_all = fid.readlines()
                lst_all = [line.strip() for line in lst_all]
        else:
            lst_all = lst

        # ToDo: Not sure if we should limit 'lst' by 'fv_msk_frames' as fv_msk_frames will no longer be part of the training except for some metric evaluation while training
        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        fv_msk_frames = os.listdir(self._fv_msk_dir)
        fv_msk_frames = [frame.split(".")[0] for frame in fv_msk_frames]
        fv_msk_frames_exist_map = {entry: True for entry in fv_msk_frames}  # This is to speed-up the dataloader
        lst = [entry for entry in lst if entry in fv_msk_frames_exist_map]
        lst_all = [entry for entry in lst_all if entry in fv_msk_frames_exist_map]
        lst_all_frame_idx_map = {entry: idx for idx, entry in enumerate(lst_all)}

        # Remove the corner scene elements so that they can satisfy the window constraint
        # if self.window is not none and the train split is considered
        if self.window > 0:
            # lst_filt = []
            lst_filt = [entry for entry in lst
                        if (((len(lst_all) - lst_all_frame_idx_map[entry]) > self.window) and (lst_all_frame_idx_map[entry] >= self.window))
                        and ((lst_all[lst_all_frame_idx_map[entry] - self.window].split(";")[0] == entry.split(";")[0]) and (lst_all[lst_all_frame_idx_map[entry] + self.window].split(";")[0] == entry.split(";")[0]))]
            # for entry in lst:
            #     scene_name = entry.split(";")[0]
            #     # frame_index = lst_all.index(entry)
            #     frame_index = lst_all_frame_idx_map[entry]
            #     if ((len(lst_all) - frame_index) <= self.window) or (frame_index < self.window):
            #         continue
            #
            #     # Check the extremes of the window to see if the images still belong to the same scene
            #     left_scene_name = lst_all[frame_index - self.window].split(";")[0]
            #     right_scene_name = lst_all[frame_index + self.window].split(";")[0]
            #
            #     if (left_scene_name == scene_name) and (right_scene_name == scene_name):
            #         lst_filt.append(entry)
            lst = lst_filt
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        bev_meta = bev_metadata["meta"]
        bev_images = [img_desc for img_desc in bev_metadata["images"] if img_desc["id"] in lst]
        fv_meta = fv_metadata["meta"]
        fv_images = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst]

        # Check for inconsistency due to inconsistencies in the input files or dataset
        bev_images_ids = [bev_img["id"] for bev_img in bev_images]
        fv_images_ids = [fv_img["id"] for fv_img in fv_images]
        assert set(bev_images_ids) == set(fv_images_ids) and len(bev_images_ids) == len(fv_images_ids), 'Inconsistency between fv_images and bev_images detected'

        if lst_all is not None:
            bev_images_all = [img_desc for img_desc in bev_metadata['images'] if img_desc['id'] in lst_all]
            fv_images_all = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst_all]
        else:
            bev_images_all, fv_images_all = None, None

        return bev_meta, bev_images, bev_images_all, fv_meta, fv_images, fv_images_all, img_map

    def _find_index(self, list, key, value):
        for i, dic in enumerate(list):
            if dic[key] == value:
                return i
        return None

    def _load_item(self, item_idx):
        # Find the index of the element in the list containing all elements
        all_idx = self._find_index(self._fv_images_all, "id", self._fv_images[item_idx]['id'])
        if all_idx is None:
            raise IOError("Required index not found!")

        if self.window > 0:
            left = all_idx - self.window
            right = all_idx + self.window
            bev_img_desc_list = self._bev_images_all[left:right+1]
            fv_img_desc_list = self._fv_images_all[left:right+1]
        else:
            bev_img_desc_list = [self._bev_images[item_idx]]
            fv_img_desc_list = [self._fv_images[item_idx]]

        scene, frame_id = self._bev_images[item_idx]["id"].split(";")

        # Get the RGB file names
        img_file = [os.path.join(self.kitti_root_dir, self._img_map["front"]["{}.png".format(bev_img_desc['id'])])
                    for bev_img_desc in bev_img_desc_list]

        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Scene: {}, Frame: {}".format(scene, frame_id))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        bev_msk_file = [os.path.join(self._bev_msk_dir, "{}.png".format(bev_img_desc['id']))
                        for bev_img_desc in bev_img_desc_list]
        bev_msk = [Image.open(msk) for msk in bev_msk_file]

        # [torch.rot90(pseudo_map, k=1, dims=[0, 1]) for pseudo_map in bev_combined_supervision]

        bev_plabel_path = os.path.join(self._bev_plabel_dir, "{}.png".format(bev_img_desc_list[len(bev_img_desc_list) // 2]['id']))
        # Load the plabel
        bev_plabel_file = [bev_plabel_path]
        to_pil_image = T.ToPILImage()
        if self.use_offline_plabels and os.path.exists(bev_plabel_path):
            # In contrast to BEV gt msks, this only takes the middle element
            bev_plabel = [Image.open(plabel).rotate(90, expand=True) for plabel in bev_plabel_file]
            # bev_plabel = [to_pil_image(torch.rot90(torch.load(plabel, map_location="cpu"), k=1, dims=[0,1]).type(torch.int)) for plabel in bev_plabel_file]
        else:
            bev_plabel = [to_pil_image(torch.rot90(torch.ones(size=bev_msk[0].size, dtype=torch.int32) * 255, k=1,dims=[0, 1])) for _ in bev_plabel_file]

        # Load the front mask
        fv_msk_file = [os.path.join(self._fv_msk_dir, "{}.png".format(fv_img_desc['id']))
                       for fv_img_desc in fv_img_desc_list]
        fv_msk = [Image.open(msk) for msk in fv_msk_file]

        assert len(fv_msk) == len(img), "FV Mask: {}, Img: {}".format(len(fv_img_desc_list), len(img))

        # Load the weight map
        # bev_weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(bev_img_desc['id']))
        # bev_weights_msk = cv2.imread(bev_weights_msk_file, cv2.IMREAD_UNCHANGED).astype(np.float)
        # if bev_weights_msk is not None:
        #     bev_weights_msk_combined = (bev_weights_msk[:, :, 0] + (bev_weights_msk[:, :, 1] / 10000)) * 10000
        #     bev_weights_msk_combined = [Image.fromarray(bev_weights_msk_combined.astype(np.int32))]
        # else:
        #     bev_weights_msk_combined = None
        bev_weights_msk_combined = None

        # Get the other information
        bev_cat = [bev_img_desc["cat"] for bev_img_desc in bev_img_desc_list]
        bev_iscrowd = [bev_img_desc["iscrowd"] for bev_img_desc in bev_img_desc_list]
        fv_cat = [fv_img_desc['cat'] for fv_img_desc in fv_img_desc_list]
        fv_iscrowd = [fv_img_desc['iscrowd'] for fv_img_desc in fv_img_desc_list]
        fv_intrinsics = [fv_img_desc["cam_intrinsic"] for fv_img_desc in fv_img_desc_list]
        ego_pose = [fv_img_desc['ego_pose'] for fv_img_desc in fv_img_desc_list]  # This loads the cam0 pose

        # Get the ids of all the frames
        frame_ids = [bev_img_desc["id"] for bev_img_desc in bev_img_desc_list]

        return img, bev_msk, bev_plabel, fv_msk, bev_weights_msk_combined, bev_cat, bev_iscrowd, \
               fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, frame_ids

    @property
    def fv_categories(self):
        """Category names"""
        return self._fv_meta["categories"]

    @property
    def fv_num_categories(self):
        """Number of categories"""
        return len(self.fv_categories)

    @property
    def fv_num_stuff(self):
        """Number of "stuff" categories"""
        return self._fv_meta["num_stuff"]

    @property
    def fv_num_thing(self):
        """Number of "thing" categories"""
        return self.fv_num_categories - self.fv_num_stuff

    @property
    def bev_categories(self):
        """Category names"""
        return self._bev_meta["categories"]

    @property
    def bev_num_categories(self):
        """Number of categories"""
        return len(self.bev_categories)

    @property
    def bev_num_stuff(self):
        """Number of "stuff" categories"""
        return self._bev_meta["num_stuff"]

    @property
    def bev_num_thing(self):
        """Number of "thing" categories"""
        return self.bev_num_categories - self.bev_num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._fv_meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._fv_meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._fv_images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._fv_images]

    @property
    def dataset_name(self):
        return "Kitti360"

    def __len__(self):
        return len(self._fv_images)

    def __getitem__(self, item):
        img, bev_msk, bev_plabel, fv_msk, bev_weights_msk, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, bev_plabel=bev_plabel, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                             bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd, fv_intrinsics=fv_intrinsics,
                             ego_pose=ego_pose)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        for m in fv_msk:
            m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class BEVWaymoDataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _BEV_PLABEL_DIR = "bev_plabel_dynamic_ws3"
    _FV_MSK_DIR = "front_msk_seam"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _BEV_METADATA_FILE = "metadata_ortho.bin"
    _FV_METADATA_FILE = "metadata_top.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform, window=0, use_offline_plabels=False):
        super(BEVWaymoDataset, self).__init__()
        self.seam_root_dir = seam_root_dir  # Directory of seamless data
        self.dataset_root_dir = dataset_root_dir  #  Directory of the KITTI360 data
        self.split_name = split_name
        self.transform = transform
        self.window = window  # Single-sided window count. The number of image samples is [0 to i + window]. Only one side is needed for non-depth training, so why waste the images?
        self.rgb_cameras = ['top']
        self.use_offline_plabels = use_offline_plabels

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVWaymoDataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVWaymoDataset._BEV_MSK_DIR, BEVWaymoDataset._BEV_DIR)
        self._bev_plabel_dir = os.path.join(seam_root_dir, BEVWaymoDataset._BEV_PLABEL_DIR, BEVWaymoDataset._BEV_DIR)
        self._fv_msk_dir = os.path.join(seam_root_dir, BEVWaymoDataset._FV_MSK_DIR, "top")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVWaymoDataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVWaymoDataset._LST_DIR)

        # Load meta-data and split
        self._fv_meta, self._fv_images, self._fv_images_all, self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        # with open(os.path.join(self.seam_root_dir, BEVWaymoDataset._BEV_METADATA_FILE), "rb") as fid:
        #     bev_metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self.seam_root_dir, BEVWaymoDataset._FV_METADATA_FILE), 'rb') as fid:
            fv_metadata = umsgpack.unpack(fid, encoding="utf-8")

        # Read the files for this split
        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        # Get all the frames in the train dataset. This will be used for generating samples for temporal consistency.
        if self.split_name == "train":
            with open(os.path.join(self._lst_dir, "{}_all.txt".format(self.split_name)), 'r') as fid:
                lst_all = fid.readlines()
                lst_all = [line.strip() for line in lst_all]
        else:
            lst_all = lst

        # Remove elements from lst if they are not in _FRONT_MSK_DIR.
        # This is critical as we need FV supervision during training! This is not needed during test though
        # if self.split_name == "train":
        fv_msk_frames = os.listdir(self._fv_msk_dir)
        fv_msk_frames = [frame.split(".")[0] for frame in fv_msk_frames]
        fv_msk_frames_exist_map = {entry: True for entry in fv_msk_frames}  # This is to speed-up the dataloader
        lst = [entry for entry in lst if entry in fv_msk_frames_exist_map]
        lst_all = [entry for entry in lst_all if entry in fv_msk_frames_exist_map]
        lst_all_frame_idx_map = {entry: idx for idx, entry in enumerate(lst_all)}

        # Remove the corner scene elements so that they can satisfy the window constraint
        # if self.window is not none and the train split is considered
        if self.window > 0:
            lst_filt = [entry for entry in lst
                        if (((len(lst_all) - lst_all_frame_idx_map[entry]) > self.window) and (lst_all_frame_idx_map[entry] >= self.window))
                        and ((lst_all[lst_all_frame_idx_map[entry] - self.window].split("-")[2] == entry.split("-")[2]) and (lst_all[lst_all_frame_idx_map[entry] + self.window].split("-")[2] == entry.split("-")[2]))]
            lst = lst_filt
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        # bev_meta = bev_metadata["meta"]
        # bev_images = [img_desc for img_desc in bev_metadata["images"] if img_desc["id"] in lst]
        fv_meta = fv_metadata["meta"]
        fv_images = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst]

        # Check for inconsistency due to inconsistencies in the input files or dataset
        # if self.split_name == "train":
            # bev_images_ids = [bev_img["id"] for bev_img in bev_images]
        fv_images_ids = [fv_img["id"] for fv_img in fv_images]
            # assert set(bev_images_ids) == set(fv_images_ids) and len(bev_images_ids) == len(fv_images_ids), 'Inconsistency between fv_images and bev_images detected'

        if lst_all is not None:
            # bev_images_all = [img_desc for img_desc in bev_metadata['images'] if img_desc['id'] in lst_all]
            fv_images_all = [img_desc for img_desc in fv_metadata['images'] if img_desc['id'] in lst_all]
        else:
            bev_images_all, fv_images_all = None, None

        return fv_meta, fv_images, fv_images_all, img_map

    def _find_index(self, list, key, value):
        for i, dic in enumerate(list):
            if dic[key] == value:
                return i
        return None

    def _load_item(self, item_idx):
        # Find the index of the element in the list containing all elements
        # if self.split_name == "train":
        all_idx = self._find_index(self._fv_images_all, "id", self._fv_images[item_idx]['id'])
        if all_idx is None:
            raise IOError("Required index not found!")

        if self.window > 0:
            left = all_idx
            right = all_idx + self.window
            # bev_img_desc_list = self._bev_images_all[left:right+1]
            fv_img_desc_list = self._fv_images_all[left:right+1]
        else:
            # bev_img_desc_list = [self._bev_images[item_idx]]
            fv_img_desc_list = [self._fv_images[item_idx]]
            # fv_img_desc_list =

        frame, location, sequence = self._fv_images[item_idx]["id"].split("-")

        # Get the RGB file names
        img_file = [os.path.join(self.dataset_root_dir, self._img_map["top"]["{}.png".format(fv_img_desc['id'])])
                    for fv_img_desc in fv_img_desc_list]
        img_file = ["{}.jpg".format(f.split(".")[0]) for f in img_file]
        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Frame: {}, Location: {}, Sequence: {}".format(frame, location, sequence))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        # bev_msk_file = [os.path.join(self._bev_msk_dir, "{}.png".format(bev_img_desc['id'])) for bev_img_desc in bev_img_desc_list]
        # bev_msk = [Image.open(msk) for msk in bev_msk_file]
        bev_msk = None
        bev_plabel = None

        # Load the plabel
        # bev_plabel_path = os.path.join(self._bev_plabel_dir, "{}.png".format(bev_img_desc_list[0]['id']))
        # bev_plabel_file = [bev_plabel_path]
        # to_pil_image = T.ToPILImage()
        # if self.use_offline_plabels and os.path.exists(bev_plabel_path):
        #     bev_plabel = [Image.open(plabel).rotate(90, expand=True) for plabel in bev_plabel_file]
        # else:
        #     bev_plabel = [to_pil_image(torch.rot90(torch.ones(size=bev_msk[0].size, dtype=torch.int32) * 255, k=1, dims=[0, 1])) for _ in bev_plabel_file]

        # Load the front mask. We can load dummy values for the test split.
        fv_msk_file = [os.path.join(self._fv_msk_dir, "{}.png".format(fv_img_desc['id']))
                       for fv_img_desc in fv_img_desc_list]
        fv_msk = [Image.open(msk) for msk in fv_msk_file]

        assert len(fv_msk) == len(img), "FV Mask: {}, Img: {}".format(len(fv_img_desc_list), len(img))

        # Load the weight map
        # bev_weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(bev_img_desc['id']))
        # bev_weights_msk = cv2.imread(bev_weights_msk_file, cv2.IMREAD_UNCHANGED).astype(np.float)
        # if bev_weights_msk is not None:
        #     bev_weights_msk_combined = (bev_weights_msk[:, :, 0] + (bev_weights_msk[:, :, 1] / 10000)) * 10000
        #     bev_weights_msk_combined = [Image.fromarray(bev_weights_msk_combined.astype(np.int32))]
        # else:
        #     bev_weights_msk_combined = None
        bev_weights_msk_combined = None

        # Get the other information
        # bev_cat = [bev_img_desc["cat"] for bev_img_desc in bev_img_desc_list]
        # bev_iscrowd = [bev_img_desc["iscrowd"] for bev_img_desc in bev_img_desc_list]
        bev_cat = None
        bev_iscrowd = None
        fv_cat = [fv_img_desc["cat"] for fv_img_desc in fv_img_desc_list]
        fv_iscrowd = [fv_img_desc["iscrowd"] for fv_img_desc in fv_img_desc_list]
        fv_intrinsics = [fv_img_desc["cam_intrinsic"] for fv_img_desc in fv_img_desc_list]
        ego_pose = [fv_img_desc['cam_pose'] for fv_img_desc in fv_img_desc_list]

        # Change the axes of the ego pose to match the camera coordinates
        T_adapt = np.zeros((4, 4), dtype=np.float32)
        T_adapt[3, 3] = 1
        T_adapt[0, 2] = 1
        T_adapt[1, 0] = -1
        T_adapt[2, 1] = -1
        ego_pose = [np.array(pose, dtype=np.float32) for pose in ego_pose]
        ego_pose = [np.matmul(pose, T_adapt) for pose in ego_pose]
        ego_pose = [pose.tolist() for pose in ego_pose]

        # Get the ids of all the frames
        frame_ids = [fv_img_desc["id"] for fv_img_desc in fv_img_desc_list]

        return img, bev_msk, bev_plabel, fv_msk, bev_weights_msk_combined, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd,\
               fv_intrinsics, ego_pose, frame_ids

    @property
    def fv_categories(self):
        """Category names"""
        return self._fv_meta["categories"]

    @property
    def fv_num_categories(self):
        """Number of categories"""
        return len(self.fv_categories)

    @property
    def fv_num_stuff(self):
        """Number of "stuff" categories"""
        return self._fv_meta["num_stuff"]

    @property
    def fv_num_thing(self):
        """Number of "thing" categories"""
        return self.fv_num_categories - self.fv_num_stuff

    @property
    def bev_categories(self):
        """Category names"""
        return []
        # return self._bev_meta["categories"]

    @property
    def bev_num_categories(self):
        """Number of categories"""
        return 0
        # return len(self.bev_categories)

    @property
    def bev_num_stuff(self):
        """Number of "stuff" categories"""
        return 0
        # return self._bev_meta["num_stuff"]

    @property
    def bev_num_thing(self):
        """Number of "thing" categories"""
        return self.bev_num_categories - self.bev_num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._fv_meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._fv_meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        # if self.split_name == "train":
        return [img_desc["size"] for img_desc in self._fv_images]
        # else:
        #     return [img_desc["size"] for img_desc in self._bev_images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._fv_images]

    @property
    def dataset_name(self):
        return "Waymo"

    def __len__(self):
        # if self.split_name == "train":
        return len(self._fv_images)
        # else:
        #     return len(self._bev_images)

    def __getitem__(self, item):
        img, bev_msk, bev_plabel, fv_msk, bev_weights_msk, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, bev_plabel=bev_plabel, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                             bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd, fv_intrinsics=fv_intrinsics,
                             ego_pose=ego_pose)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        if bev_msk is not None:
            for m in bev_msk:
                m.close()
        if fv_msk is not None:
            for m in fv_msk:
                m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)

class BEVWaymoDepthDataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _BEV_PLABEL_DIR = "bev_plabel"
    _FV_MSK_DIR = "front_msk_seam"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _BEV_METADATA_FILE = "metadata_ortho.bin"
    _FV_METADATA_FILE = "metadata_top.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform, window=0):
        super(BEVWaymoDepthDataset, self).__init__()
        self.seam_root_dir = seam_root_dir  # Directory of seamless data
        self.dataset_root_dir = dataset_root_dir  #  Directory of the KITTI360 data
        self.split_name = split_name
        self.transform = transform
        self.window = window  # Single-sided window count. The number of image samples is [0 to i + window]. Only one side is needed for non-depth training, so why waste the images?
        self.rgb_cameras = ['top']

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVWaymoDataset._IMG_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVWaymoDataset._LST_DIR)

        # Load meta-data and split
        self._bev_meta, self._bev_images, self._bev_images_all, self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVWaymoDataset._BEV_METADATA_FILE), "rb") as fid:
            bev_metadata = umsgpack.unpack(fid, encoding="utf-8")

        # Read the files for this split
        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        # Get all the frames in the train dataset. This will be used for generating samples for temporal consistency.
        if self.split_name == "train":
            with open(os.path.join(self._lst_dir, "{}_all.txt".format(self.split_name)), 'r') as fid:
                lst_all = fid.readlines()
                lst_all = [line.strip() for line in lst_all]
        else:
            lst_all = lst

        lst_all_frame_idx_map = {entry: idx for idx, entry in enumerate(lst_all)}

        # Remove the corner scene elements so that they can satisfy the window constraint
        # if self.window is not none and the train split is considered
        if self.window > 0:
            lst_filt = [entry for entry in lst
                        if (((len(lst_all) - lst_all_frame_idx_map[entry]) > self.window) and (lst_all_frame_idx_map[entry] >= self.window))
                        and ((lst_all[lst_all_frame_idx_map[entry] - self.window].split("-")[2] == entry.split("-")[2]) and (lst_all[lst_all_frame_idx_map[entry] + self.window].split("-")[2] == entry.split("-")[2]))]
            lst = lst_filt
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        bev_meta = bev_metadata["meta"]
        bev_images = [img_desc for img_desc in bev_metadata["images"] if img_desc["id"] in lst]


        if lst_all is not None:
            bev_images_all = [img_desc for img_desc in bev_metadata['images'] if img_desc['id'] in lst_all]
        else:
            bev_images_all = None

        return bev_meta, bev_images, bev_images_all, img_map

    def _find_index(self, list, key, value):
        for i, dic in enumerate(list):
            if dic[key] == value:
                return i
        return None

    def _load_item(self, item_idx):
        # Find the index of the element in the list containing all elements
        all_idx = self._find_index(self._bev_images_all, "id", self._bev_images[item_idx]['id'])
        if all_idx is None:
            raise IOError("Required index not found!")

        if self.window > 0:
            left = all_idx - self.window
            right = all_idx + self.window
            bev_img_desc_list = self._bev_images_all[left:right+1]
        else:
            bev_img_desc_list = [self._bev_images[item_idx]]

        frame, location, sequence = self._bev_images[item_idx]["id"].split("-")

        # Get the RGB file names
        img_file = [os.path.join(self.dataset_root_dir, self._img_map["top"]["{}.png".format(bev_img_desc['id'])])
                    for bev_img_desc in bev_img_desc_list]
        img_file = ["{}.jpg".format(f.split(".")[0]) for f in img_file]
        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Frame: {}, Location: {}, Sequence: {}".format(frame, location, sequence))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        fv_intrinsics = [bev_img_desc["cam_intrinsic"] for bev_img_desc in bev_img_desc_list]
        ego_pose = [bev_img_desc['cam_pose'] for bev_img_desc in bev_img_desc_list]

        # Get the ids of all the frames
        frame_ids = [bev_img_desc["id"] for bev_img_desc in bev_img_desc_list]

        return img, None, None, None, None, None, None, None, fv_intrinsics, ego_pose, frame_ids

    @property
    def fv_categories(self):
        """Category names"""
        return self._fv_meta["categories"]

    @property
    def fv_num_categories(self):
        """Number of categories"""
        return len(self.fv_categories)

    @property
    def fv_num_stuff(self):
        """Number of "stuff" categories"""
        return self._fv_meta["num_stuff"]

    @property
    def fv_num_thing(self):
        """Number of "thing" categories"""
        return self.fv_num_categories - self.fv_num_stuff

    @property
    def bev_categories(self):
        """Category names"""
        return self._bev_meta["categories"]

    @property
    def bev_num_categories(self):
        """Number of categories"""
        return len(self.bev_categories)

    @property
    def bev_num_stuff(self):
        """Number of "stuff" categories"""
        return self._bev_meta["num_stuff"]

    @property
    def bev_num_thing(self):
        """Number of "thing" categories"""
        return self.bev_num_categories - self.bev_num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._fv_meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._fv_meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._fv_images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._fv_images]

    @property
    def dataset_name(self):
        return "Waymo"

    def __len__(self):
        return len(self._fv_images)

    def __getitem__(self, item):
        img, bev_msk, fv_msk, bev_weights_msk, bev_cat, bev_iscrowd, fv_cat, fv_iscrowd, fv_intrinsics, ego_pose, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, fv_msk=fv_msk, bev_weights_msk=bev_weights_msk, bev_cat=bev_cat,
                             bev_iscrowd=bev_iscrowd, fv_cat=fv_cat, fv_iscrowd=fv_iscrowd, fv_intrinsics=fv_intrinsics,
                             ego_pose=ego_pose)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        for m in fv_msk:
            m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)




if __name__ == "__main__":
    from po_bev_unsupervised.data.misc import iss_collate_fn
    from po_bev_unsupervised.data.sampler import DistributedARBatchSampler
    from po_bev_unsupervised.utils.sequence import pad_packed_images

    # KITTI
    # train_tf = BEVTransform(shortest_size=384,
    #                         longest_max_size=1408,
    #                         rgb_mean=[0.485, 0.456, 0.406],
    #                         rgb_std=[0.229, 0.224, 0.225],
    #                         front_resize=(384, 1408),
    #                         bev_crop=[768, 704])
    # train_db = BEVKitti360Dataset(seam_root_dir="/home/gosalan/data/kitti360_bev_seam_poses",
    #                               dataset_root_dir='/home/gosalan/data/kitti360_dataset',
    #                               split_name="train",
    #                               transform=train_tf,
    #                               window=2)

    # Waymo
    train_tf = BEVTransform(shortest_size=1280,
                            longest_max_size=1920,
                            rgb_mean=[0.485, 0.456, 0.406],
                            rgb_std=[0.229, 0.224, 0.225],
                            front_resize=(448, 768),
                            bev_crop=[896, 768])
    train_db = BEVWaymoDepthDataset(seam_root_dir="/home/gosalan/data/waymo_bev_seam",
                                  dataset_root_dir='/home/gosalan/data/waymo_dataset',
                                  split_name="train",
                                  transform=train_tf,
                                  window=5)
    train_sampler = DistributedARBatchSampler(train_db, 3, 1, 0, True)
    train_dl = torch.utils.data.DataLoader(train_db,
                                           # batch_size=dl_config.getint('train_batch_size'),
                                           batch_sampler=train_sampler,
                                           collate_fn=iss_collate_fn,
                                           pin_memory=True,
                                           num_workers=1)

    print("BEV Num Things", train_db.bev_num_thing)
    print("BEV Num Stuff", train_db.bev_num_stuff)
    print("FV Num Things", train_db.fv_num_thing)
    print("FV Num Stuff", train_db.fv_num_stuff)

    for i, sample in enumerate(train_dl):
        img = sample["img"]
        # po_msk = sample["bev_msk"]
        front_msk = sample['fv_msk']

        front_msk, _ = pad_packed_images(front_msk)

        # sem_class_mask = sample["sem_class_mask"]

        # print("Img", img.shape)
        # print("Depth GT", depth_gt.shape)
        # print(torch.max(depth_gt))
        # print(torch.min(depth_gt))
        # print("Semantics GT", sem_gt.shape)

        # print(torch.max(po_msk))
        # print("Sem Class Mask", sem_class_mask.shape)


