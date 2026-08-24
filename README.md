# UFDR: Uncertainty-Guided Feature Distribution Refinement for Unsupervised Medical Image Anomaly Detection




- PUCL = Pairwise Uncertainty-guided Curriculum Learning
- TGDR = Trajectory-Guided Decoder Regulation
- RCA = Re-parameterized calibration attention



## 目录

```text
UFDR-github-minimal/
├── configs/ufdr.yaml       # 唯一的示例配置
├── scripts/train.py        # 训练入口
├── scripts/test.py         # 图像级评估入口
├── ufdr/                   # 模型、机制、数据与引擎
├── tests/                  # CPU 友好的单元/集成测试
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

## 安装

建议在独立 Python 虚拟环境中，从包根目录执行：

```bash
python -m pip install -r requirements.txt
```

生产编码器通过下列 provider 模块构建 DINOv3 ConvNeXt-Tiny：

```text
lightly_train._models.dinov3.dinov3_src.hub.backbones
```


```bash
python -c "from lightly_train._models.dinov3.dinov3_src.hub.backbones import dinov3_convnext_tiny; print('provider OK')"
```

另外请自行取得与该 provider 兼容的 DINOv3 ConvNeXt-Tiny 权重，并放在配置指定的位置。默认配置期望：

```text
weights/dinov3_convnext_tiny_lvd1689m.pth
```



## 数据布局

`data.root` 必须具有如下文件夹结构；目录内可以继续嵌套子目录：

```text
data/example/
├── train/
│   └── normal/
├── val/
│   └── normal/
└── test/
    ├── normal/
    └── anomaly/
```

支持 `.png`、`.jpg`、`.jpeg`、`.bmp`、`.tif`、`.tiff`（扩展名不区分大小写）。训练和验证只读取正常图像；测试必须同时包含正常和异常图像。输入会缩放为 `image_size × image_size` 并归一化到 `[-1, 1]`。`channels` 可取 1 或 3；单通道输入会在生产编码器入口复制成三通道。真实模型的空间尺寸需能被 32 整除。

## 配置

默认配置是 `configs/ufdr.yaml`。当配置文件位于 `configs/` 时，`data.root`、`model.weights` 和 `train.output_dir` 的相对路径均相对于包根目录解析，而不是相对于当前 shell 工作目录；因此脚本可从其他目录调用。

| 字段 | 含义 |
| --- | --- |
| `seed`, `device` | 随机种子与 `cpu` / `cuda` / `cuda:N` 设备 |
| `data.root` | 上述数据根目录 |
| `data.image_size`, `data.channels` | 方形输入尺寸与通道数 |
| `data.batch_size`, `data.workers` | batch 大小与 DataLoader worker 数 |
| `model.encoder`, `model.weights` | 固定编码器类型与外部权重路径 |
| `model.aux_view` | 固定为 `rot180` |
| `model.freeze_encoder_epochs` | 冻结 DINO backbone 的前若干 epoch；projection 仍训练 |
| `model.cosine_weight`, `model.mse_weight` | 三层特征重建损失权重 |
| `pucl.temperature`, `pucl.weight`, `pucl.eps` | PUCL 温度、总损失权重与数值稳定项 |
| `pucl.label_mode`, `pucl.group_size` | PUCL 正样本标签策略及分组大小 |
| `tgdr.window_size` | 训练/验证轨迹滑动窗口 |
| `tgdr.base_l2_lambda`, `tgdr.max_lambda` | 自适应正则下界系数与上限 |
| `tgdr.target`, `tgdr.reliability` | 固定为 `decoder1` 与 `corr_gap` |
| `train.epochs` | 训练 epoch 数 |
| `train.lr_encoder`, `train.lr_projection`, `train.lr_decoder` | DINO backbone、encoder projection、decoder 学习率；RCA 随 decoder 使用 `lr_decoder` |
| `train.output_dir` | checkpoint 输出目录 |

PUCL 的 `label_mode` 语义如下：

- `class`：直接使用数据集类别标签；当前训练/验证文件夹中的正常样本标签均为 0。
- `group`：按一个 batch 内的顺序和 `group_size` 将相邻样本划为一组；每组共享标签。
- `instance`：batch 内每个样本使用唯一标签，仅该样本的两个视图构成同实例正对。

## 训练与测试

从包根目录启动训练：

```bash
python scripts/train.py --config configs/ufdr.yaml
```

训练会在 `train.output_dir` 下原子写入最佳验证 checkpoint `best.pt`。终端 JSON 包含 `history`、`best_val_loss` 和 `best_checkpoint`；checkpoint 只保存格式版本、epoch 与模型状态，不写入本机绝对数据路径。

训练优化策略与原实现保持一致：使用 `AdamW`（`weight_decay=1e-4`），DINO backbone、encoder projection 和其余 decoder/RCA 参数分别使用三档学习率；前 10% epoch 执行 `10% linear warmup`，之后使用 `CosineAnnealingLR`（`eta_min=1e-7`）。每次反向传播后执行 `clip_grad_norm_=0.5`。本包循环内部 epoch 从 0 开始，但在每个 epoch 末通过 `epoch + 1` 对齐原实现的 1-based warmup/cosine 更新时序。

训练 history、最佳 checkpoint 选择和反向传播使用包含 TGDR 正则的 total loss；TGDR 轨迹更新只观察未含自身正则项的 `loss_base`（缺失时回退到 total loss），从而避免正则反馈回路。

测试命令为：

```bash
python scripts/test.py --config configs/ufdr.yaml --checkpoint outputs/ufdr/best.pt
```

测试输出为 JSON，包含图像级 ROC `auc`、`average_precision` 和 `num_samples`。此最小数据接口没有像素掩码，因此不计算像素级指标。

可随时查看脚本实际参数：

```bash
python scripts/train.py --help
python scripts/test.py --help
```

## 数据流

同一个 shared DINOv3 ConvNeXt-Tiny encoder 分别编码 orig 图像和其 rot180 视图。两条分支使用 two independent decoders；每条分支在第 2、3 层特征各使用一个 RCA，共 4 个 RCA。两个 decoder 分别重建前三层编码特征，产生 cosine + MSE feature reconstruction loss。PUCL 使用两视图最深层池化特征构造课程式对比目标；TGDR 根据近期训练/验证损失轨迹调整 L2 强度，且只正则 decoder1。

推理时，两条分支都以编码/解码特征的余弦差异生成异常图。rot180 分支的异常图先反向旋转回 orig 坐标，再与主分支平均融合；最终图像分数是融合图的空间均值。

## CPU smoke 与真实运行

无需 provider、DINOv3 权重或真实数据即可执行 CPU 测试：

```bash
python -m pytest -q
```

测试通过轻量注入模型和假的 provider 检查组件、数据、checkpoint 与 CLI 契约；这只验证代码路径，不代表研究性能。真实 UFDR 使用双 DINO 前向和两个 ResNet-50 风格 decoder，建议在准备兼容权重后在 CUDA GPU 上运行，并依据显存调小 `data.batch_size`。如明确选择 CPU，请把 `device` 改为 `cpu`，但完整训练会很慢。

## 发布与许可提示

PUCL 中适配代码的第三方归属和 MIT 许可全文见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。外部 DINOv3/provider 不属于本包。本目录没有替你决定整个项目的发布许可证；手动上传前，请根据你对其余代码的授权情况选择项目级 `LICENSE`，第三方声明不能替代项目级许可。
