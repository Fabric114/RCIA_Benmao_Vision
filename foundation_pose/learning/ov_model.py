from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WEIGHTS_ROOT = Path(__file__).resolve().parents[1] / "weights"
REFINE_RUN = "2023-10-28-18-33-37"
SCORE_RUN = "2024-01-11-20-02-45"
PRECISIONS = ("fp32", "int8")
OV_PRECISIONS = ("fp32", "fp16", "int8")
EXPORT_CMD = "python scripts/export_foundationpose_openvino.py"
QUANTIZE_CMD = "python scripts/quantize_foundationpose_openvino.py"


def _ir_path(weights_root, run, stem, precision):
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision!r}, expected one of {PRECISIONS}")
    return Path(weights_root) / run / "openvino" / f"{stem}_{precision}.xml"


def ov_precision_to_ir(ov_precision: str) -> tuple[str, str]:
    """把部署参数 ov_precision 映射成 (IR 精度, OpenVINO INFERENCE_PRECISION_HINT)。

    fp16 不生成新的 IR：读 FP32 IR，编译时把推理精度提示设为 f16。
    """
    if ov_precision == "fp32":
        return "fp32", "f32"
    if ov_precision == "fp16":
        return "fp32", "f16"
    if ov_precision == "int8":
        return "int8", "f32"
    raise ValueError(f"unknown ov_precision: {ov_precision!r}, expected one of {OV_PRECISIONS}")


def refine_ir_path(weights_root, precision: str = "fp32") -> Path:
    return _ir_path(weights_root, REFINE_RUN, "refine", precision)


def score_ir_path(weights_root, precision: str = "fp32") -> Path:
    return _ir_path(weights_root, SCORE_RUN, "score", precision)


class RefineExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, A, B):
        n = self.net
        x = n.encodeA(torch.cat([A, B], dim=0))
        a, b = torch.chunk(x, 2, dim=0)
        ab = n.encodeAB(torch.cat([a, b], dim=1))
        ab = n.pos_embed(ab.flatten(2).permute(0, 2, 1))
        return n.trans_head(ab).mean(dim=1), n.rot_head(ab).mean(dim=1)


class ScoreExportWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, A, B):
        n = self.net
        x = n.encoderA(torch.cat([A, B], dim=0))
        a, b = torch.chunk(x, 2, dim=0)
        ab = n.encoderAB(torch.cat([a, b], dim=1))
        ab = n.pos_embed(ab.flatten(2).permute(0, 2, 1))
        ab, _ = n.att(ab, ab, ab)
        feats = ab.mean(dim=1).unsqueeze(0)
        x, _ = n.att_cross(feats, feats, feats)
        return n.linear(x)[..., 0]


class OVModel:
    """每个输入 batch 大小单独 reshape 成静态 shape 编译，并缓存编译结果。

    动态 batch 的 IR 在 OpenVINO 2024.6 CPU 上有错误：先推理过 N=1，再推理 N=20，
    score 输出偏差约 1.4。静态 shape 的结果与 torch 一致。
    """

    def __init__(self, xml_path, output_keys, device: str = "CPU", num_threads: int | None = None,
                 cache_dir: str | None = None, precision_hint: str = "f32"):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            if "int8" in xml_path.name:
                hint = f"请先运行 `{QUANTIZE_CMD}` 生成 INT8 IR。"
            else:
                hint = f"请先运行 `{EXPORT_CMD}` 导出 FP32 IR。"
            raise FileNotFoundError(f"OpenVINO IR 不存在: {xml_path}。{hint}")
        import openvino as ov

        self.xml_path = xml_path
        self.device = device
        self.output_keys = tuple(output_keys)
        self.core = ov.Core()
        self.config = {"INFERENCE_PRECISION_HINT": precision_hint, "PERFORMANCE_HINT": "LATENCY"}
        if num_threads is not None:
            self.config["INFERENCE_NUM_THREADS"] = int(num_threads)
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self.config["CACHE_DIR"] = str(cache_dir)
        # IR 只读一次保存在内存中，每个 shape 编译时用 clone 做 reshape，不再重复读盘。
        self._raw_model = self.core.read_model(str(self.xml_path))
        self._compiled = {}

    def _compiled_for(self, shape_a, shape_b):
        key = (tuple(shape_a), tuple(shape_b))
        if key not in self._compiled:
            model = self._raw_model.clone()
            inputs = model.inputs
            model.reshape({inputs[0]: list(shape_a), inputs[1]: list(shape_b)})
            self._compiled[key] = self.core.compile_model(model, self.device, self.config)
        return self._compiled[key]

    def warmup(self, batch_sizes, hw=(160, 160)):
        """对每个 batch 大小预先编译静态图，并用全零输入推理一次，避免首次推理时才编译。"""
        h, w = hw
        for n in batch_sizes:
            compiled = self._compiled_for((n, 6, h, w), (n, 6, h, w))
            zeros = np.zeros((n, 6, h, w), dtype=np.float32)
            compiled({0: zeros, 1: zeros})
        return self

    @property
    def compiled_batch_sizes(self) -> list:
        """已编译的 batch 大小。只在 device="GPU" 时有内容。"""
        if self.device != "GPU":
            return []
        return sorted({key[0][0] for key in self._compiled})

    def __call__(self, A: torch.Tensor, B: torch.Tensor, **unused) -> dict:
        compiled = self._compiled_for(A.shape, B.shape)
        res = compiled({0: A.detach().float().cpu().numpy(), 1: B.detach().float().cpu().numpy()})
        return {
            key: torch.from_numpy(np.array(res[compiled.outputs[i]], dtype=np.float32)).to(A.device)
            for i, key in enumerate(self.output_keys)
        }
