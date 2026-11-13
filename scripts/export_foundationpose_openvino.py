import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_DIR = PROJECT_ROOT / "foundation_pose"
if str(FOUNDATIONPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from learning.models.refine_network import RefineNet  # noqa: E402
from learning.models.score_network import ScoreNetMultiPair  # noqa: E402
from learning.ov_model import (  # noqa: E402
    REFINE_RUN,
    SCORE_RUN,
    WEIGHTS_ROOT,
    RefineExportWrapper,
    ScoreExportWrapper,
    refine_ir_path,
    score_ir_path,
)


def _load_net(weights_root, run, build):
    run_dir = Path(weights_root) / run
    cfg = OmegaConf.load(run_dir / "config.yml")
    net = build(cfg)
    ckpt = torch.load(run_dir / "model_best.pth", map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    net.load_state_dict(ckpt)
    return net.eval()


def export_all(weights_root, out_root):
    import openvino as ov

    torch.backends.mha.set_fastpath_enabled(False)
    refine = _load_net(weights_root, REFINE_RUN, lambda c: RefineNet(cfg=c, c_in=c["c_in"]))
    score = _load_net(weights_root, SCORE_RUN, lambda c: ScoreNetMultiPair(cfg=c, c_in=c["c_in"]))

    example = (torch.rand(4, 6, 160, 160), torch.rand(4, 6, 160, 160))
    shapes = [[-1, 6, 160, 160], [-1, 6, 160, 160]]
    with torch.no_grad():
        refine_m = ov.convert_model(RefineExportWrapper(refine), example_input=example, input=shapes)
        score_m = ov.convert_model(ScoreExportWrapper(score), example_input=example, input=shapes)

    refine_out = refine_ir_path(out_root, "fp32")
    score_out = score_ir_path(out_root, "fp32")
    refine_out.parent.mkdir(parents=True, exist_ok=True)
    score_out.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(refine_m, str(refine_out), compress_to_fp16=False)
    ov.save_model(score_m, str(score_out), compress_to_fp16=False)
    return refine_out, score_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights_root", default=str(WEIGHTS_ROOT))
    p.add_argument("--out_root", default=None)
    args = p.parse_args()
    out_root = args.out_root or args.weights_root
    for path in export_all(args.weights_root, out_root):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
