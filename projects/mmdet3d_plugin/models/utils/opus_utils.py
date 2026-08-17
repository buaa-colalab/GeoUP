import torch

def encode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    points[..., 1] = (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    points[..., 2] = (points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return points


def decode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points

def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    # Assumes v0.17.1+ style or whatever StreamPETR uses. 
    # OPUS utils.py checked VERSION.name. Here we assume standard rotation matrix.
    # [cos, -sin, 0]
    # [sin, cos, 0]
    # [0, 0, 1]
    # But transpose(0, 1) means we are building the transpose? 
    # OPUS code:
    # if VERSION.name == 'v0.17.1':
    #     rot_mat_T = torch.stack([
    #         rot_cos, -rot_sin, zeros,
    #         rot_sin, rot_cos, zeros,
    #         zeros, zeros, ones,
    #     ]).transpose(0, 1).reshape(-1, 3, 3)
    # else:
    #     rot_mat_T = torch.stack([
    #         rot_cos, rot_sin, zeros,
    #         -rot_sin, rot_cos, zeros,
    #         zeros, zeros, ones,
    #     ]).transpose(0, 1).reshape(-1, 3, 3)
    
    # Let's use the 'else' case which seems to be the default/newer logic in OPUS code provided.
    
    rot_mat_T = torch.stack([
        rot_cos, rot_sin, zeros,
        -rot_sin, rot_cos, zeros,
        zeros, zeros, ones,
    ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)
    
    return points

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
