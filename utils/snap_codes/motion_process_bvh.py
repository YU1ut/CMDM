import sys

sys.path.append("../")

from os.path import join as pjoin
from scipy.ndimage import gaussian_filter1d

from .common.skeleton import Skeleton
import numpy as np
import os
from .common.quaternion import *

import torch
from tqdm import tqdm

from einops import rearrange, repeat
from collections import defaultdict

contact_joint_names = [
    "L_foot0001_bind_JNT",
    "L_foot0002_bind_JNT",
    "R_foot0001_bind_JNT",
    "R_foot0002_bind_JNT",
]

face_joint_names = [
    "R_legUpper0001_bind_JNT",
    "L_legUpper0001_bind_JNT",
    "R_clavicle_bind_JNT",
    "L_clavicle_bind_JNT",
]

joints_dict = {}
kinematic_tree = []


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# rot_data (B, seq_len, joint_num * 6)
# ric_data (B, seq_len, joint_num * 3)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis Rotation from Rotation Velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang / 2, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # (vx, vy, vz, r) - > (cos(r/2), vx * sin(r/2), vy * sin(r/2), vz * sin(r/2))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    """Get Root Positions"""
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]

    #     print(torch.sum(r_pos**2, axis=-1)[:100])
    """Add Y-axis Rotation to Root Positions"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    #     print(torch.sum(r_pos**2, axis=-1)[:100])

    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# rot_data (B, seq_len, joint_num * 6)
# ric_data (B, seq_len, joint_num * 3)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_bvh_from_rot(data, joints_num, skeleton, keep_shape=True):
    b, l, _ = data.shape
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1
    end_indx = start_indx + joints_num * 6
    # print(data.shape)
    cont6d_params = data[..., start_indx:end_indx].reshape(b, l, joints_num, 6)
    quat_params = cont6d_to_quaternion(cont6d_params)
    # print(quat_params.shape)

    # print(r_rot_quat.shape, quat_params.shape)
    quat_params = rearrange(quat_params, "b l j d -> (b l) j d")
    r_rot_quat = repeat(r_rot_quat, "b l d -> (b l) j d", j=quat_params.shape[1])
    r_pos = rearrange(r_pos, "b l d -> (b l) d")

    global_quats = qmul(qinv(r_rot_quat), quat_params)
    # global_quats = qmul(qinv(r_rot_quat)[:, np.newaxis].repeat(1,joints_num, 1), quat_params)
    local_quats = skeleton.global_to_local_quat(global_quats)

    if keep_shape:
        global_quats = rearrange(global_quats, "(b l) j d -> b l j d", b=b)
        local_quats = rearrange(local_quats, "(b l) j d -> b l j d", b=b)
        r_pos = rearrange(r_pos, "(b l) d -> b l d", b=b)
    return global_quats, local_quats, r_pos


def recover_pos_from_rot(data, joints_num, skeleton):
    # print(data.shape)
    b, l, _ = data.shape
    global_quats, _, r_pos = recover_bvh_from_rot(
        data, joints_num, skeleton, keep_shape=False
    )
    global_pos = skeleton.fk_global_quat(global_quats, r_pos)
    global_pos = rearrange(global_pos, "(b l) j d -> b l j d", b=b)
    return global_pos


def recover_pos_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    start_indx = 1 + 2 + 1 + joints_num * 6
    end_indx = start_indx + joints_num * 3
    positions = data[..., start_indx:end_indx]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # print(positions.shape)

    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )

    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # '''Concate root and joints'''
    # positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def extract_kinematic_chains(animation):
    kinematic_chains = []
    chain = []
    for i, parent_i in enumerate(animation.parents):
        if parent_i != i - 1:
            kinematic_chains.append(chain)
            chain = [parent_i, i]
        else:
            chain.append(i)

    kinematic_chains.append(chain)
    return kinematic_chains


if __name__ == "__main__":
    src_root = "/mnt/local-disk/omni_motion_dataset/raw_data/"
    tgt_root = "/mnt/local-disk/omni_motion_dataset/raw_data/renamed_feats"
    # folders = os.listdir(src_root)
    folders = ["renamed_bvhs"]
    length_dict = defaultdict(int)
    # print(folders)
    for folder in folders:
        print(folder)
        fnames = os.listdir(pjoin(src_root, folder))
        target_path = pjoin(tgt_root, folder)
        os.makedirs(target_path, exist_ok=True)
        num_frame = 0
        for name in tqdm(fnames):
            # target_path = pjoin(target_root, "m_"+name)
            try:
                # if dac
                # if os.path.exists(pjoin(target_path,"%s.npy"%name[:-4])):
                #     continue
                data = process_bvh_motion(
                    pjoin(src_root, folder, name),
                    30,
                    30,
                    feet_thre=0.11,
                    shift_one_frame=True,
                )
                np.save(pjoin(target_path, "%s.npy" % name[:-4]), data)
                num_frame += len(data)
                length_dict[name.split("_")[0]] += len(data)
            except Exception as e:
                print(name)
                print(e)

        print(
            "Total duration: %04fh, average duration: %04fs"
            % (num_frame / 30 / 3600, num_frame / 30 / len(fnames))
        )
        # for key, value in length_dict:
        #     print("Total duration of %s: %04fh"%(key, value/30/3600))
