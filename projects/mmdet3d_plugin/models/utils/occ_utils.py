import torch


def sparse2dense(indices, value, dense_shape, empty_value=0):
    B, N = indices.shape[:2]

    batch_index = torch.arange(B, device=value.device).unsqueeze(1).expand(B, N)
    dense = torch.full(
        [B] + dense_shape,
        empty_value,
        device=value.device,
        dtype=value.dtype)
    dense[batch_index, indices[..., 0], indices[..., 1], indices[..., 2]] = value

    mask = torch.zeros([B] + dense_shape[:3], dtype=torch.bool, device=value.device)
    mask[batch_index, indices[..., 0], indices[..., 1], indices[..., 2]] = True

    return dense, mask
