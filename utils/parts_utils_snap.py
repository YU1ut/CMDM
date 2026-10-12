"""SnapMoGen 296-feature motion split into six Part-TMR body parts."""

import torch


def whole2parts_snap_batch(motion: torch.Tensor) -> list[torch.Tensor]:
    if len(motion.shape) == 2:
        motion = motion.unsqueeze(0)

    assert (
        len(motion.shape) == 3
    ), "Motion data must be 3D: (batch_size, nframes, feature_dim)"

    batch_size, nframes, _ = motion.shape
    joints_num = 24

    s = 0
    e = 4
    root_data = motion[:, :, s:e]
    s = e
    e = e + joints_num * 6
    rot_data = motion[:, :, s:e]
    s = e
    e = e + joints_num * 3
    ric_data = motion[:, :, s:e]
    s = e
    e = e + joints_num * 3
    local_vel = motion[:, :, s:e]
    s = e
    e = e + 4
    feet = motion[:, :, s:e]

    device = motion.device
    # BVH joint order: pelvis 15, left leg 16-19, right leg 20-23.
    r_l_idx = torch.tensor([15, 20, 21, 22, 23], device=device, dtype=torch.int64)
    l_l_idx = torch.tensor([15, 16, 17, 18, 19], device=device, dtype=torch.int64)
    b_idx = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64)
    # BVH joint order: left arm 7-10, right arm 11-14; joint 3 is shared.
    r_a_idx = torch.tensor([3, 11, 12, 13, 14], device=device, dtype=torch.int64)
    l_a_idx = torch.tensor([3, 7, 8, 9, 10], device=device, dtype=torch.int64)

    ric_data = ric_data.reshape(batch_size, nframes, -1, 3)
    rot_data = rot_data.reshape(batch_size, nframes, -1, 6)
    local_vel = local_vel.reshape(batch_size, nframes, -1, 3)

    root_data = torch.cat([root_data, local_vel[:, :, 0, :]], dim=-1)
    r_l = torch.cat(
        [
            ric_data[:, :, r_l_idx, :],
            rot_data[:, :, r_l_idx, :],
            local_vel[:, :, r_l_idx, :],
        ],
        dim=-1,
    )
    l_l = torch.cat(
        [
            ric_data[:, :, l_l_idx, :],
            rot_data[:, :, l_l_idx, :],
            local_vel[:, :, l_l_idx, :],
        ],
        dim=-1,
    )
    b = torch.cat(
        [
            ric_data[:, :, b_idx, :],
            rot_data[:, :, b_idx, :],
            local_vel[:, :, b_idx, :],
        ],
        dim=-1,
    )
    r_a = torch.cat(
        [
            ric_data[:, :, r_a_idx, :],
            rot_data[:, :, r_a_idx, :],
            local_vel[:, :, r_a_idx, :],
        ],
        dim=-1,
    )
    l_a = torch.cat(
        [
            ric_data[:, :, l_a_idx, :],
            rot_data[:, :, l_a_idx, :],
            local_vel[:, :, l_a_idx, :],
        ],
        dim=-1,
    )

    root = root_data
    r_leg = torch.cat([r_l.reshape(batch_size, nframes, -1), feet[:, :, 2:]], dim=-1)
    l_leg = torch.cat([l_l.reshape(batch_size, nframes, -1), feet[:, :, :2]], dim=-1)
    backbone = b.reshape(batch_size, nframes, -1)
    r_arm = r_a.reshape(batch_size, nframes, -1)
    l_arm = l_a.reshape(batch_size, nframes, -1)

    return [root, r_leg, l_leg, backbone, r_arm, l_arm]
