"""Ascend fix lowering regression tests."""

import re

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from testing.ascend._ir import calls


def device_source(kernel):
    """Lower explicit Cube/Vector synchronization without automatic scheduling."""
    with tvm.transform.PassContext(config={"tl.enable_auto_schedule": False}):
        return tilelang.lower(kernel, target="ascend", enable_device_compile=False).kernel_source


def fix_drain(m, n, split, *, dtype="bfloat16", scale=0.08838834764831845, source_row=0):
    """Build a Cube product followed by a single or M-split FIX drain."""
    copied_rows = m - source_row
    rows = copied_rows // 2 if split else copied_rows
    vector_count = 2 if split else 1
    pitch = m // 16

    @T.prim_func
    def kernel(
        a: T.Tensor((m, 128), "bfloat16"),
        b: T.Tensor((n, 128), "bfloat16"),
        out: T.Tensor((m, n), dtype),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, 128), "bfloat16")
            b_l1 = T.alloc_l1((n, 128), "bfloat16")
            a_l0 = T.alloc_l0a((m, 128), "bfloat16")
            b_l0 = T.alloc_l0b((n, 128), "bfloat16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((rows, n), dtype)
            with T.Cube():
                T.copy(a[0:m, 0:128], a_l1)
                T.copy(b[0:n, 0:128], b_l1)
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1, a_l0)
                T.copy(b_l1, b_l0)
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=True)
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                if split:
                    T.dual_copy(acc[source_row:m, 0:n], ub[0:rows, 0:n], fix_scale=scale)
                else:
                    T.copy(acc[source_row:m, 0:n], ub[0:rows, 0:n], fix_scale=scale, sub_blockid=0, l0_pitch=pitch)
                T.ascend_cross_core_set_flag(2, "PIPE_FIX", 8)
            with T.Vector(vector=vector_count):
                T.ascend_cross_core_wait_flag(2, "PIPE_MTE3", 8)
                if split:
                    T.dual_copy(ub[0:rows, 0:n], out[0:copied_rows, 0:n])
                else:
                    T.copy(ub[0:rows, 0:n], out[0:copied_rows, 0:n])
                T.ascend_pipe_barrier("PIPE_ALL")

    return kernel


@pytest.mark.parametrize("m,n", [(128, 128), (256, 64)])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("dtype,quant_pre", [("float16", 32), ("bfloat16", 34), ("float32", 15)])
def test_scaled_fix(m, n, split, dtype, quant_pre):
    """Check scalar prequantization and the single-destination routing of each half."""
    cpp = device_source(fix_drain(m, n, split, dtype=dtype))
    assert cpp.count("asc_set_l0c_copy_prequant(") == 1
    assert re.search(r"asc_set_l0c_copy_prequant\((?:\(uint64_t\))?(?:0x3db50000|1035272192)", cpp)
    lines = [line for line in cpp.splitlines() if "asc_copy_l0c2ub(" in line]
    assert len(lines) == (2 if split else 1)
    for subblock, line in enumerate(lines):
        assert f", {subblock}, static_cast<asc_dual_dst_mode>(0)" in line
        assert f"static_cast<asc_quant_mode>({quant_pre})" in line
    if split:
        assert re.search(rf"\+ {m * 8}\b", lines[1])


@pytest.mark.parametrize("dtype,quant_pre", [("float16", 1), ("bfloat16", 16), ("float32", 0)])
def test_unscaled_fix(dtype, quant_pre):
    """Keep the original cast or unscaled FP32 mode when no scale is supplied."""
    cpp = device_source(fix_drain(128, 128, False, dtype=dtype, scale=None))
    assert "asc_set_l0c_copy_prequant(" not in cpp
    assert f"static_cast<asc_quant_mode>({quant_pre})" in cpp


@pytest.mark.parametrize("scale", [1e100, 2**2048, 2**-130])
def test_fix_scale_requires_normal_fp32(scale):
    """Reject scalars that overflow or become subnormal in the FIX descriptor."""
    with pytest.raises(Exception, match="normal FP32 scale"):
        device_source(fix_drain(128, 128, False, scale=scale))


def test_scaled_fix_rejects_nonzero_source_origin():
    """A split drain must start at the origin of its L0C tile."""
    with pytest.raises(Exception, match="zero-origin"):
        device_source(fix_drain(128, 128, True, source_row=32))


def test_fix_scale_rejects_scalar_copy_outside_fix_memory():
    """A regular scalar copy has no FIX path and must reject scaling."""
    with pytest.raises(Exception, match="FP32 L0C to FP16, BF16, or FP32 UB matrix drain"):

        @T.prim_func
        def kernel(src: T.Tensor((1,), "float32"), dst: T.Tensor((1,), "float32")):
            with T.Kernel(1):
                T.copy(src[0], dst[0], fix_scale=0.5)


def test_fix_scale_annotation_preserved_on_scalar_drain():
    """The encoded annotation takes precedence and keeps a scalar copy as a tile op."""

    @T.prim_func
    def kernel():
        with T.Kernel(1):
            acc = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float16")
            T.copy(acc[0, 0], ub[0, 0], annotations={"fix_scale_bits": T.uint64(0x3F000000)}, fix_scale=0.0)

    (copy,) = calls(kernel, "tl.tileop.ascend_copy")
    assert int(copy.annotations["fix_scale_bits"]) == 0x3F000000
