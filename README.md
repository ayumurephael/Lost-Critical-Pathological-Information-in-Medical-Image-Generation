# 无标注 OCT 病灶候选区域估计代码库

本代码库用于解决课题第一步的核心问题：在没有人工病灶标注的情况下，如何从术前 OCT 图像 `A_i` 中估计可能的病灶区域，并输出 `lesion mask / bbox`。

我们的数据是成对的：

- `A_i`：术前 OCT，受白内障影响，噪声和模糊更明显。
- `B_i`：同一只眼的术后样或术后 OCT，更清晰。
- 训练时可以使用 `A_i` 和 `B_i`。
- 推理时只能使用 `A_i`。
- 数据集中不知道病灶有多少类，也不知道每张图是否真的有病灶。

一个最重要的约束是：本项目只允许使用 `数据集\1-30`，代码会主动拒绝 `数据集\1-60`。

## 一句话理解

训练阶段先借助更清晰的术后图像 `B_i` 找到“像病灶的候选区域”，把这个区域当成伪标签转移给同一眼的术前图像 `A_i`，再训练一个只看 `A_i` 的分割模型。推理阶段就不再需要 `B_i`，输入一张术前 OCT 即可输出 mask 和 bbox。

## 与原 MUIS 论文的关系

原论文 `Multiscale Unsupervised Retinal Edema Area Segmentation in OCT Images` 要解决的是“没有像素级标注时，如何做视网膜水肿区域分割”。它的核心思想可以拆成三步：

1. 不要求人工画病灶边界，而是先用无监督方法从 OCT 图像中学习图像级病变模式。
2. 用聚类或类别激活图 CAM 找出最可能影响图像级病变模式的区域。
3. 把这些区域后处理成伪 mask，再训练一个真正的分割网络。

本代码库保留了这个思想，但根据我们的任务做了迁移：

- 原论文只有 OCT 图像本身；我们有成对图像 `A_i, B_i`，其中 `B_i` 更清晰，所以 `B_i` 是训练阶段天然的“弱 teacher”。
- 原论文关注水肿；我们的数据不知道病灶种类数，因此不能假设只有一种病灶。本代码使用多尺度异常响应、无监督 teacher CAM 和质量过滤来估计“候选病灶区域”，不需要事先知道类别数。
- 原论文最终仍要训练分割模型；这里也训练分割模型，但模型输入只能是术前图像 `A_i`，这样才能满足真实推理场景。

也就是说，这不是简单照搬论文代码，而是把论文的“无标注候选区域构造”思想迁移到了“术后清晰图辅助术前病灶定位”的成对 OCT 场景。

## 总体流程

### 第 1 步：建立成对数据清单

`build_manifest.py` 会扫描 `数据集\1-30`，找到每一对术前和术后 OCT，生成一个 CSV 清单。后续所有训练和推理都只读这个清单。

### 第 2 步：用 `B_i` 生成伪病灶标签

默认推荐使用 `run_experiment.py`。它会在术后图像 `B_i` 上计算多尺度异常分数，包括暗区、亮区、局部对比、LoG 结构响应，并结合术前图像 `A_i` 的结构支持。然后它会自动搜索阈值、最小面积、形态学半径和成对支持权重，选择质量最稳定的一组参数。

这一步输出的是伪标签，不是人工真值。伪标签的作用是让后面的 student 网络知道“大概哪里值得关注”。

### 第 3 步：可选的 MUIS/DCCS teacher CAM 路径

为了更完整地对应原论文，本仓库还提供了 `train_teacher.py` 和 `make_teacher_cam_pseudo.py`：

- `train_teacher.py` 在术后图像 `B_i` 上训练一个 MUIS/DCCS 风格的无监督聚类 teacher。
- `make_teacher_cam_pseudo.py` 从 teacher 的类别激活图 CAM 中生成伪 mask。

这条路径更接近原论文的“聚类 teacher 到 CAM 再到 mask”。默认实验推荐先使用多尺度确定性伪标签，因为它在小数据、无人工标签和成对 OCT 场景下更稳定；CAM 路径适合在实验室 GPU 上做对照实验或融合实验。

### 第 4 步：训练只看 `A_i` 的 student 分割网络

`train_student.py` 使用术前图像 `A_i` 作为输入，使用第 2 步或第 3 步得到的伪 mask 作为训练目标。训练完成后，模型推理时不需要术后图像 `B_i`。

为了避免本机 CPU 烟测版过于简化，现在默认配置已经换成面向实验室算力的研究版：

- 输入分辨率默认 `768 x 496`，不再使用低分辨率快速实验设置。
- 分割网络为残差注意力 U-Net，带 SE 模块、ASPP 多尺度上下文和深监督。
- 损失函数使用 focal loss、Dice loss、Tversky loss 的组合，更适合病灶区域很小、前景背景极不平衡的情况。
- 自动根据伪 mask 前景比例计算 `pos_weight`。
- 默认启用 AMP 混合精度、多 worker 数据加载、Cosine 学习率调度和梯度裁剪。
- 验证时自动搜索多个阈值，保存伪标签 IoU 最好的 checkpoint。

### 第 5 步：推理输出 mask 和 bbox

`infer.py` 输入训练好的 student checkpoint 和术前图像清单，输出：

- `masks/`：每张术前 OCT 的预测 mask。
- `overlays/`：mask 叠加在原图上的可视化图。
- `predictions.csv`：每张图的 bbox、面积比例、概率统计和使用的阈值。

## 文件结构

```text
Codes\1
├─ lesion_candidate
│  ├─ build_manifest.py          # 扫描 1-30 数据集，生成成对图像清单
│  ├─ data.py                    # 数据集读取、配对、划分训练验证集
│  ├─ preprocess.py              # OCT 预处理、归一化、bbox、overlay 等工具
│  ├─ pseudo.py                  # 多尺度无标注伪病灶候选区域生成
│  ├─ make_pseudo.py             # 用指定参数批量生成伪标签
│  ├─ run_experiment.py          # 搜索伪标签参数，推荐默认入口
│  ├─ models.py                  # MUIS teacher、CAM 模块、student 分割网络
│  ├─ train_teacher.py           # 可选：训练 MUIS/DCCS 风格 teacher
│  ├─ make_teacher_cam_pseudo.py # 可选：用 teacher CAM 生成伪标签
│  ├─ train_student.py           # 训练术前 A-only student 分割网络
│  └─ infer.py                   # 推理，输出 mask 和 bbox
├─ tests
│  └─ test_manifest.py           # 基础数据清单测试
├─ requirements.txt
├─ pyproject.toml
└─ README.md
```

## 环境准备

建议在实验室 GPU 服务器上使用 Python 3.10 或更高版本，并安装 CUDA 版 PyTorch。进入本目录后安装依赖：

```powershell
cd Codes\1
python -m pip install -r requirements.txt
```

如果服务器上需要单独安装 CUDA 版 PyTorch，请按实验室 CUDA 版本安装对应 wheel。这个仓库没有固定死 torch 版本，是为了兼容不同 GPU 服务器环境。

## 推荐完整实验命令

以下命令都在 `Codes\1` 目录中运行。

### 1. 生成 manifest

```powershell
python -m lesion_candidate.build_manifest --dataset-root ..\..\数据集\1-30 --out artifacts\manifest_pairs_1_30.csv
```

这一步会生成 `artifacts\manifest_pairs_1_30.csv`。如果误传 `1-60`，代码会直接报错并停止。

### 2. 搜索最佳伪标签参数

```powershell
python -m lesion_candidate.run_experiment --manifest artifacts\manifest_pairs_1_30.csv --out-dir artifacts\research_grid
```

研究版默认会搜索全部样本，而不是只抽少量样本。输出中最重要的是：

```text
artifacts\research_grid\best\pseudo_manifest.csv
```

这个文件就是后续训练 student 的伪标签清单。

### 3. 训练 A-only student

```powershell
python -m lesion_candidate.train_student --manifest artifacts\research_grid\best\pseudo_manifest.csv --out-dir artifacts\student_research --epochs 200 --batch-size 8 --base 64 --width 768 --height 496 --pos-weight auto --num-workers 8 --compile-model
```

显存不足时，优先把 `--batch-size` 从 8 降到 4 或 2，不建议降低分辨率。训练完成后主要看：

```text
artifacts\student_research\student_best.pt
artifacts\student_research\train_log.csv
artifacts\student_research\student_report.md
artifacts\student_research\val_predictions
```

注意：`val_best_iou` 是模型和伪标签之间的 IoU，不是人工标注 IoU。它只能说明模型是否学会了伪标签规律，不能直接等价为临床真实性能。

### 4. 对术前图像推理

```powershell
python -m lesion_candidate.infer --checkpoint artifacts\student_research\student_best.pt --manifest artifacts\manifest_pairs_1_30.csv --out-dir artifacts\student_research_infer
```

输出结果在：

```text
artifacts\student_research_infer\masks
artifacts\student_research_infer\overlays
artifacts\student_research_infer\predictions.csv
```

`predictions.csv` 中的 bbox 字段就是每张术前 OCT 的候选病灶框。

## 可选：原论文风格 teacher CAM 实验

如果你想更贴近 MUIS 原论文，可以先训练 teacher：

```powershell
python -m lesion_candidate.train_teacher --manifest artifacts\manifest_pairs_1_30.csv --out-dir artifacts\muis_teacher --epochs 300 --batch-size 64 --image-size 192 --aux-size 384 --dim-zc 8 --dim-zs 64 --base 64 --num-workers 8 --compile-model
```

然后用 teacher CAM 生成伪标签：

```powershell
python -m lesion_candidate.make_teacher_cam_pseudo --checkpoint artifacts\muis_teacher\teacher_last.pt --manifest artifacts\manifest_pairs_1_30.csv --out-dir artifacts\teacher_cam_pseudo --target-width 768 --target-height 496 --cam-mode pred
```

再用 CAM 伪标签训练 student：

```powershell
python -m lesion_candidate.train_student --manifest artifacts\teacher_cam_pseudo\pseudo_manifest.csv --out-dir artifacts\student_teacher_cam --epochs 200 --batch-size 8 --base 64 --width 768 --height 496 --pos-weight auto --num-workers 8 --compile-model
```

建议把默认多尺度伪标签和 teacher CAM 伪标签都训练一版 student，对比 overlay 图、预测面积分布和验证伪 IoU。如果二者关注区域一致，可信度更高；如果差异很大，需要回看原图判断是哪一路更合理。

## 为什么这样能满足“推理时只能用 A”

训练时使用 `B_i` 只是为了构造伪标签。student 看到的输入始终是 `A_i`。训练结束后，checkpoint 中只保存 student 网络参数，推理脚本也只读取 `pre_path`，也就是术前 OCT。

所以最终部署形式是：

```text
术前 OCT A_i -> student 分割网络 -> lesion mask -> bbox
```

## 如何判断结果好不好

没有人工标注时不能只看一个数值，建议同时看这些证据：

1. `overlays/` 中的红色区域是否落在 OCT 结构异常处，而不是图像边缘、黑背景或文字噪声。
2. `predictions.csv` 中 `pred_area_fraction` 是否大体合理。若大量图接近 0，说明过保守；若大量图很大，说明过泛化。
3. 同一病例相邻切片的结果是否连续。真实病灶一般不会在相邻切片中完全随机跳动。
4. 默认多尺度伪标签 student 和 teacher CAM student 是否经常关注同一区域。
5. 有条件时，请医生抽查少量 overlay。即使只抽查 30 到 50 张，也能快速发现方向性问题。

## 当前实现相对早期简化版的升级

本版本已经把早期为了 CPU 快速验证而做的简化替换为研究版默认设置：

- 伪标签分辨率从低分辨率升级到 `768 x 496`。
- 伪标签参数搜索默认遍历全部样本，而不是只抽少量样本。
- student 网络从轻量 U-Net 升级为残差注意力 ASPP U-Net。
- 训练轮数默认提升到 200，teacher 默认提升到 300。
- 损失函数从简单 BCE/Dice 升级为 focal、Dice、Tversky 混合损失。
- 加入深监督、自动前景权重、多阈值验证、AMP、Cosine 调度和 `torch.compile` 开关。
- 新增 teacher CAM 伪标签生成入口，更完整地覆盖原论文思路。

## 重要提醒

本项目输出的是“病灶候选区域”，不是经过医生确认的最终诊断结果。它适合作为后续医学图像生成、病灶保持约束、候选框筛选或人工复核的第一步。真正用于论文定量评价时，最好再加入少量人工标注切片作为外部验证集。
