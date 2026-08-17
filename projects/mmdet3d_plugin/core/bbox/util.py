import torch

def normalize_bbox(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

def normalize_bbox_center(bbox, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    bbox_center = bbox[..., :3]
    bbox_other = bbox[..., 3:]
    new_bbox_center = bbox_center.clone()
    new_bbox_center[...,0:1] = bbox_center[...,0:1] - pc_range[0]
    new_bbox_center[...,1:2] = bbox_center[...,1:2] - pc_range[1]
    new_bbox_center[...,2:3] = bbox_center[...,2:3] - pc_range[2]
    factor = bbox_center.new_tensor([patch_w, patch_h, patch_z])
    normalized_bbox_center = new_bbox_center / factor
    normalized_bbox = torch.cat((normalized_bbox_center, bbox_other), dim=-1)
    return normalized_bbox

def normalize_far_bbox_center(bbox, pc_range):
    if pc_range[4] < 150:
        return bbox
    else:
        factor = pc_range[4] // 50
        bbox_center = bbox[..., :2]
        bbox_other = bbox[..., 2:]
        bbox_center = bbox_center / factor
        normalized_bbox = torch.cat((bbox_center, bbox_other), dim=-1)
        return normalized_bbox
