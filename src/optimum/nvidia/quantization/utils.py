import torch


def unpack_int32_into_int8(w_packed, transpose: bool = False, center: bool = False):
    # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
    if transpose:
        w_packed = w_packed.T

    w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
    w_unpacked = torch.zeros(w_packed_int4x2.shape[0], w_packed_int4x2.shape[1] * 2, dtype=torch.int8)
    w_unpacked[:, ::2] = w_packed_int4x2 % 16
    w_unpacked[:, 1::2] = w_packed_int4x2 // 16

    if transpose:
        w_unpacked = w_unpacked.T

    if center:
        w_unpacked -= 8

    return w_unpacked.contiguous()


def pack_int8_to_int4(src: torch.ByteTensor) -> torch.ByteTensor:
    (d_out, d_in) = src.size()
    out = torch.zeros(d_out, (d_in + 1) // 2, dtype=torch.int8, requires_grad=False)

    # for packed_idx in range(out.numel()):
    #     lsb = src[]

    return out