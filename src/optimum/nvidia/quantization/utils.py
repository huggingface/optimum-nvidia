import torch


def unpack_int32_into_int8(w_packed, transpose: bool = False, center: bool = False):
    # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
    if transpose:
        w_packed = w_packed.T

    w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
    w_unpacked = torch.zeros(w_packed_int4x2.shape[0], w_packed_int4x2.shape[1] * 2, dtype=torch.int8)
    w_unpacked[:, ::2] = w_packed_int4x2 % 16
    w_unpacked[:, 1::2] = w_packed_int4x2 // 16

    if center:
        w_unpacked -= 8

    if transpose:
        w_unpacked = w_unpacked.T

    return w_unpacked.contiguous()