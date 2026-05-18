import codecs as cs
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


def whole2parts(motion, mode="t2m", window_size=None):
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    if len(motion.shape) == 2:
        motion = motion.unsqueeze(0)  # Add a batch dimension if missing

    assert (
        len(motion.shape) == 3
    ), "Motion data must be 3D: (batch_size, nframes, feature_dim)"

    batch_size, nframes, _ = motion.shape

    # motion
    if mode == "t2m":
        # 263-dims motion is actually an augmented motion representation
        joints_num = 22
        s = 0  # start
        e = 4  # end
        root_data = motion[:, :, s:e]  # [batch_size, seq_len, 4]
        s = e
        e = e + (joints_num - 1) * 3
        ric_data = motion[:, :, s:e]  # [batch_size, seq_len, (joints_num-1)*3]
        s = e
        e = e + (joints_num - 1) * 6
        rot_data = motion[:, :, s:e]  # [batch_size, seq_len, (joints_num-1) *6]
        s = e
        e = e + joints_num * 3
        local_vel = motion[:, :, s:e]  # [batch_size, seq_len, joints_num*3]
        s = e
        e = e + 4
        feet = motion[:, :, s:e]  # [batch_size, seq_len, 4]

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(torch.int64)  # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(torch.int64)  # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(torch.int64)  # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(torch.int64)  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(torch.int64)  # left arm

        if window_size is not None:
            assert nframes == window_size

        # Reshape and split data according to joint indices
        ric_data = ric_data.reshape(
            batch_size, nframes, -1, 3
        )  # (batch_size, nframes, joints_num - 1, 3)
        rot_data = rot_data.reshape(
            batch_size, nframes, -1, 6
        )  # (batch_size, nframes, joints_num - 1, 6)
        local_vel = local_vel.reshape(
            batch_size, nframes, -1, 3
        )  # (batch_size, nframes, joints_num, 3)

        root_data = torch.cat(
            [root_data, local_vel[:, :, 0, :]], dim=-1
        )  # (batch_size, nframes, 4+3=7)
        R_L = torch.cat(
            [
                ric_data[:, :, R_L_idx - 1, :],
                rot_data[:, :, R_L_idx - 1, :],
                local_vel[:, :, R_L_idx, :],
            ],
            dim=-1,
        )  # (batch_size, nframes, 4, 3+6+3=12)
        L_L = torch.cat(
            [
                ric_data[:, :, L_L_idx - 1, :],
                rot_data[:, :, L_L_idx - 1, :],
                local_vel[:, :, L_L_idx, :],
            ],
            dim=-1,
        )  # (batch_size, nframes, 4, 3+6+3=12)
        B = torch.cat(
            [
                ric_data[:, :, B_idx - 1, :],
                rot_data[:, :, B_idx - 1, :],
                local_vel[:, :, B_idx, :],
            ],
            dim=-1,
        )  # (batch_size, nframes, 5, 3+6+3=12)
        R_A = torch.cat(
            [
                ric_data[:, :, R_A_idx - 1, :],
                rot_data[:, :, R_A_idx - 1, :],
                local_vel[:, :, R_A_idx, :],
            ],
            dim=-1,
        )  # (batch_size, nframes, 5, 3+6+3=12)
        L_A = torch.cat(
            [
                ric_data[:, :, L_A_idx - 1, :],
                rot_data[:, :, L_A_idx - 1, :],
                local_vel[:, :, L_A_idx, :],
            ],
            dim=-1,
        )  # (batch_size, nframes, 5, 3+6+3=12)

        # Reshape parts into their final form
        Root = root_data
        R_Leg = torch.cat(
            [R_L.reshape(batch_size, nframes, -1), feet[:, :, 2:]], dim=-1
        )  # (batch_size, nframes, 50)
        L_Leg = torch.cat(
            [L_L.reshape(batch_size, nframes, -1), feet[:, :, :2]], dim=-1
        )  # (batch_size, nframes, 50)
        Backbone = B.reshape(batch_size, nframes, -1)  # (batch_size, nframes, 60)
        R_Arm = R_A.reshape(batch_size, nframes, -1)  # (batch_size, nframes, 60)
        L_Arm = L_A.reshape(batch_size, nframes, -1)  # (batch_size, nframes, 60)

    else:
        raise Exception("Invalid mode specified: {}".format(mode))

    return [Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm]


def parts2whole(parts, mode="t2m", shared_joint_rec_mode="Avg"):
    assert isinstance(parts, list)

    if mode == "t2m":
        # Parts to whole. (7, 50, 50, 60, 60, 60) ==> 263
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 22
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([2, 5, 8, 11]).to(device, dtype=torch.int64)  # right leg
        L_L_idx = torch.Tensor([1, 4, 7, 10]).to(device, dtype=torch.int64)  # left leg
        B_idx = torch.Tensor([3, 6, 9, 12, 15]).to(
            device, dtype=torch.int64
        )  # backbone
        R_A_idx = torch.Tensor([9, 14, 17, 19, 21]).to(
            device, dtype=torch.int64
        )  # right arm
        L_A_idx = torch.Tensor([9, 13, 16, 18, 20]).to(
            device, dtype=torch.int64
        )  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]

        else:
            R_L = R_Leg[..., :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(bs, nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip(
            [R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]
        ):
            # rec_ric_data[:, idx - 1, :] = part[:, :, :3]
            # rec_rot_data[:, idx - 1, :] = part[:, :, 3:9]
            # rec_local_vel[:, idx, :] = part[:, :, 9:]

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 9th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 9

        if shared_joint_rec_mode == "L_Arm":
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == "R_Arm":
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == "Backbone":
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == "Avg":
            rec_ric_data[..., idx - 1, :] = (
                L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]
            ) / 3
            rec_rot_data[..., idx - 1, :] = (
                L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]
            ) / 3
            rec_local_vel[..., idx, :] = (
                L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]
            ) / 3

        else:
            raise Exception()

        # Concate them to 263-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=1,
            )

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=2,
            )

    elif mode == "kit":

        # Parts to whole. (7, 62, 62, 48, 48, 48) ==> 251
        # we need to get root_data, ric_data, rot_data, local_vel, feet

        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:
            bs = None
            nframes = Root.shape[0]
        else:
            raise Exception()

        joints_num = 21
        device = Root.device

        rec_root_data = Root[..., :4]
        rec_feet = torch.cat([L_Leg[..., -2:], R_Leg[..., -2:]], dim=-1)

        # move the root out of belowing parts
        R_L_idx = torch.Tensor([11, 12, 13, 14, 15]).to(
            device, dtype=torch.int64
        )  # right leg
        L_L_idx = torch.Tensor([16, 17, 18, 19, 20]).to(
            device, dtype=torch.int64
        )  # left leg
        B_idx = torch.Tensor([1, 2, 3, 4]).to(device, dtype=torch.int64)  # backbone
        R_A_idx = torch.Tensor([3, 5, 6, 7]).to(device, dtype=torch.int64)  # right arm
        L_A_idx = torch.Tensor([3, 8, 9, 10]).to(device, dtype=torch.int64)  # left arm

        if bs is None:
            R_L = R_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[:, 0, :] = Root[:, 4:]

        else:
            R_L = R_Leg[..., :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[..., :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)

            # ric_data, rot_data, local_vel
            rec_ric_data = torch.zeros(bs, nframes, joints_num - 1, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_rot_data = torch.zeros(bs, nframes, joints_num - 1, 6).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel = torch.zeros(bs, nframes, joints_num, 3).to(
                device, dtype=rec_root_data.dtype
            )
            rec_local_vel[..., 0, :] = Root[..., 4:]

        for part, idx in zip(
            [R_L, L_L, B, R_A, L_A], [R_L_idx, L_L_idx, B_idx, R_A_idx, L_A_idx]
        ):

            rec_ric_data[..., idx - 1, :] = part[..., :, :3]
            rec_rot_data[..., idx - 1, :] = part[..., :, 3:9]
            rec_local_vel[..., idx, :] = part[..., :, 9:]

        # ########################
        # Choose the origin of 3-th joint, from B, R_A, L_A, or compute the mean
        # ########################
        idx = 3

        if shared_joint_rec_mode == "L_Arm":
            rec_ric_data[..., idx - 1, :] = L_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = L_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = L_A[..., 0, 9:]

        elif shared_joint_rec_mode == "R_Arm":
            rec_ric_data[..., idx - 1, :] = R_A[..., 0, :3]
            rec_rot_data[..., idx - 1, :] = R_A[..., 0, 3:9]
            rec_local_vel[..., idx, :] = R_A[..., 0, 9:]

        elif shared_joint_rec_mode == "Backbone":
            rec_ric_data[..., idx - 1, :] = B[..., 2, :3]
            rec_rot_data[..., idx - 1, :] = B[..., 2, 3:9]
            rec_local_vel[..., idx, :] = B[..., 2, 9:]

        elif shared_joint_rec_mode == "Avg":
            rec_ric_data[..., idx - 1, :] = (
                L_A[..., 0, :3] + R_A[..., 0, :3] + B[..., 2, :3]
            ) / 3
            rec_rot_data[..., idx - 1, :] = (
                L_A[..., 0, 3:9] + R_A[..., 0, 3:9] + B[..., 2, 3:9]
            ) / 3
            rec_local_vel[..., idx, :] = (
                L_A[..., 0, 9:] + R_A[..., 0, 9:] + B[..., 2, 9:]
            ) / 3

        else:
            raise Exception()

        # Concate them to 251-dims repre
        if bs is None:
            rec_ric_data = rec_ric_data.reshape(nframes, -1)
            rec_rot_data = rec_rot_data.reshape(nframes, -1)
            rec_local_vel = rec_local_vel.reshape(nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=1,
            )

        else:
            rec_ric_data = rec_ric_data.reshape(bs, nframes, -1)
            rec_rot_data = rec_rot_data.reshape(bs, nframes, -1)
            rec_local_vel = rec_local_vel.reshape(bs, nframes, -1)

            rec_data = torch.cat(
                [rec_root_data, rec_ric_data, rec_rot_data, rec_local_vel, rec_feet],
                dim=2,
            )

    else:
        raise Exception()

    return rec_data


def get_each_part_vel(parts, mode="t2m"):
    assert isinstance(parts, list)

    if mode == "t2m":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            B = Backbone.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(
                bs, nframes, 4, -1
            )  # (bs, nframes, 4, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 5, -1)  # (bs, nframes, 5, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(
                bs, nframes, -1
            )  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    elif mode == "kit":
        # Extract each part's velocity from parts representation
        Root, R_Leg, L_Leg, Backbone, R_Arm, L_Arm = parts

        if len(Root.shape) == 3:  # (bs, nframes, part_repre)
            bs = Root.shape[0]
            nframes = Root.shape[1]

        elif len(Root.shape) == 2:  # (nframes, part_repre)
            bs = None
            nframes = Root.shape[0]

        else:
            raise Exception()

        Root_vel = Root[..., 4:]
        if bs is None:
            R_L = R_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :-2].reshape(nframes, 5, -1)  # (nframes, 5, 3+6+3=12)
            B = Backbone.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(nframes, 4, -1)  # (nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, 9:].reshape(nframes, -1)
            L_Leg_vel = L_L[:, :, 9:].reshape(nframes, -1)
            Backbone_vel = B[:, :, 9:].reshape(nframes, -1)
            R_Arm_vel = R_A[:, :, 9:].reshape(nframes, -1)
            L_Arm_vel = L_A[:, :, 9:].reshape(nframes, -1)

        else:
            R_L = R_Leg[:, :, :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            L_L = L_Leg[:, :, :-2].reshape(
                bs, nframes, 5, -1
            )  # (bs, nframes, 5, 3+6+3=12)
            B = Backbone.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            R_A = R_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)
            L_A = L_Arm.reshape(bs, nframes, 4, -1)  # (bs, nframes, 4, 3+6+3=12)

            R_Leg_vel = R_L[:, :, :, 9:].reshape(
                bs, nframes, -1
            )  # (bs, nframes, nb_joints, 3) ==> (bs, nframes, vel_dim)
            L_Leg_vel = L_L[:, :, :, 9:].reshape(bs, nframes, -1)
            Backbone_vel = B[:, :, :, 9:].reshape(bs, nframes, -1)
            R_Arm_vel = R_A[:, :, :, 9:].reshape(bs, nframes, -1)
            L_Arm_vel = L_A[:, :, :, 9:].reshape(bs, nframes, -1)

        parts_vel_list = [
            Root_vel,
            R_Leg_vel,
            L_Leg_vel,
            Backbone_vel,
            R_Arm_vel,
            L_Arm_vel,
        ]

    else:
        raise Exception()

    return parts_vel_list  # [Root_vel, R_Leg_vel, L_Leg_vel, Backbone_vel, R_Arm_vel, L_Arm_vel]


from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax. From stylegan2-ADA"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
