import torch


def patch_ids_from_boundaries(boundary_mask):
    """Convert boolean boundary mask to integer patch IDs via cumsum

    boundary_mask: [B, L] boolean, True at patch start positions
    returns: [B, L] integer, each byte's patch index (0-based)
    """

    return boundary_mask.long().cumsum(dim=1) - 1


def enforce_patch_constraints(boundary_mask, min_patch, max_patch):
    """Enforce min/max patch size on boundary predictions

    Scans left-to-right: skip boundaries too close to previous,
    force boundaries when distance exceeds max_patch
    """

    B, L   = boundary_mask.shape
    device = boundary_mask.device
    result = torch.zeros_like(boundary_mask)

    for b in range(B):
        last = 0
        result[b, 0] = True

        for i in range(1, L):
            dist = i - last

            if dist >= max_patch:
                result[b, i] = True
                last          = i
            elif dist >= min_patch and boundary_mask[b, i]:
                result[b, i] = True
                last          = i

    return result


def scatter_pool(byte_embeds, patch_ids, num_patches):
    """Pool byte embeddings into patches via scatter_reduce mean

    byte_embeds: [B, L, D]
    patch_ids:   [B, L] integer, mapping bytes to patches
    num_patches: int, max number of patches (P)
    returns:     [B, P, D]
    """

    B, L, D = byte_embeds.shape
    device  = byte_embeds.device
    dtype   = byte_embeds.dtype

    patches = torch.zeros(B, num_patches, D, device=device, dtype=dtype)
    counts  = torch.zeros(B, num_patches, 1, device=device, dtype=dtype)

    idx = patch_ids.unsqueeze(-1).expand(-1, -1, D)

    patches.scatter_add_(1, idx, byte_embeds)
    counts.scatter_add_(1, patch_ids.unsqueeze(-1), torch.ones(B, L, 1, device=device, dtype=dtype))

    counts  = counts.clamp(min=1)
    patches = patches / counts

    return patches


def scatter_unpool(patch_embeds, patch_ids, num_bytes):
    """Expand patch embeddings back to byte resolution via gather

    patch_embeds: [B, P, D]
    patch_ids:    [B, L] integer, mapping bytes to patches
    num_bytes:    int, target byte length (L)
    returns:      [B, L, D]
    """

    D   = patch_embeds.shape[2]
    idx = patch_ids.unsqueeze(-1).expand(-1, num_bytes, D)

    return torch.gather(patch_embeds, 1, idx)
