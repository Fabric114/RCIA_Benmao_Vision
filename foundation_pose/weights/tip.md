# FoundationPose weights

This project does not commit the public FoundationPose network weights.

Download the weights from the official FoundationPose repository instructions:

- Official repository: https://github.com/NVlabs/FoundationPose
- Official weights folder: https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing
- Optional Hugging Face mirror used by this project's helper script: https://huggingface.co/gpue/foundationpose-weights

Expected local paths:

```text
foundation_pose/weights/2023-10-28-18-33-37/model_best.pth
foundation_pose/weights/2024-01-11-20-02-45/model_best.pth
```

You can also run:

```bash
bash scripts/download_public_weights.sh
```

## OpenVINO IR 导出

在仓库根目录、激活 `foundationpose_ros` 环境后执行：

```bash
python scripts/export_foundationpose_openvino.py
```

会生成 `2023-10-28-18-33-37/openvino/refine_fp32.xml` 和 `2024-01-11-20-02-45/openvino/score_fp32.xml`（FP32，未压缩为 FP16）。权重更新后需重新导出。

## INT8 量化（NNCF）

需要先完成上面的 FP32 导出。校准帧放在 `runtime/fp_eval/calib_frames/<名字>/`（每帧含 `color.png`、`depth.png`、`camera_info.yaml`，并用 `foundationpose_eval.py prepare` 生成 `bbox.json`），不要复用评测集 `runtime/fp_eval/frames/`。

```bash
python scripts/quantize_foundationpose_openvino.py collect --frames_root runtime/fp_eval/calib_frames --out runtime/fp_eval/calib
python scripts/quantize_foundationpose_openvino.py quantize --calib runtime/fp_eval/calib
```

会生成 `2023-10-28-18-33-37/openvino/refine_int8.xml` 和 `2024-01-11-20-02-45/openvino/score_int8.xml`。加 `--conv_only` 时只量化卷积，结果会覆盖同名文件。
