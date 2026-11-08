# OODExplore：VLM 推理状态与视觉证据联合检测

普通 Qwen2.5-VL 生成一次显式解释与答案，原图/同尺寸灰图在**同一输出、同一文本历史**上计算几何分数。复用已有 OODmethod 的 `GeometryScorer`，不重新实现 NCI/fDBD，不训练检测器，不重跑原论文文本实验。

默认：3B、ScienceQA 带图子集、100 dev / 500 test、greedy、384 tokens、fDBD k=1000。7B 仅需更换模型配置。

**交付状态：代码已构建；按用户要求未在本机运行测试、加载模型或执行实验。** 下列检查和命令供服务器执行；尚无实测效果或显存结论。

服务器首次克隆和日常 Git 同步见 [服务器部署与 Git 同步指引](docs/server-sync.md)。代码目录固定为 `/projects/OODexplore`（注意大小写），大文件根目录默认为 `/wdl/data`。

## 1. 服务器安装（Conda）

建议 Linux，使用独立的 Conda 环境（Python 3.10）。服务器目录约定：

```text
/projects/OODexplore/          # 本项目 Git 仓库
/projects/OODmethod/           # 已验证复现代码，只读引用，须另行同步
/wdl/data/
├── huggingface/              # hub 权重、数据原始下载、datasets Arrow、Xet 缓存
├── oodexplore/
│   ├── datasets/scienceqa/    # 固定子集 JSONL 与图片
│   └── runs/                 # 逐 token 结果、图表、分析
├── conda/                    # 环境与包缓存
├── cache/                    # pip、torch 缓存
└── tmp/                      # 安装/下载临时文件
```

在服务器的 OODExplore 项目根目录内，先创建并激活环境：

```bash
cd /projects/OODexplore
source scripts/server_env.sh
conda env create --prefix /wdl/data/conda/envs/oodexplore -f environment.yml
conda activate /wdl/data/conda/envs/oodexplore
```

环境文件提供 Python、pip、构建工具及 pytest；PyTorch 按服务器驱动选择后安装，其他项目依赖由 `pyproject.toml` 管理，避免重复维护两套依赖列表。

以下以 PyTorch 2.6.0 / torchvision 0.21.0 的 CUDA 12.4 wheel 为例；确认服务器驱动支持该构建后执行：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip check
python -m ood_explore show-config --config configs/qwen3b.json
```

若服务器需要其他 CUDA 构建，请使用 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/#v260) 中对应的命令，并保持 torch 与 torchvision 版本配套；这里没有把服务器的 CUDA 版本写死在 Conda 文件中。

后续每次登录服务器，先进入 `/projects/OODexplore`，执行 `source scripts/server_env.sh` 和 `conda activate /wdl/data/conda/envs/oodexplore`。环境已经创建时不必再次 `conda env create`。可用 `conda env export --no-builds > /wdl/data/oodexplore-environment.resolved.yml` 留存安装后的环境信息。

集群批处理脚本中通常需要先初始化 Conda，再激活环境：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
cd /projects/OODexplore
source scripts/server_env.sh
conda activate /wdl/data/conda/envs/oodexplore
bash scripts/run_server.sh
```

该示例要求 `conda` 已在 PATH 中；若集群通过 module 提供 Conda，请先按集群约定加载它。

本项目固定 `transformers==4.51.3`，与旧复现环境分离。**不要将 OODmethod 连同其旧 Transformers 依赖安装进本环境**：这里只按文件路径导入其原有评分模块。也不会写入 OODmethod。

`oodmethod_dir` 默认 `/projects/OODmethod`；该复现仓库须另行同步，不会随本项目自动下载。若服务器上已有其他路径，请修改 JSON；相对路径相对本项目根目录解析。`model` 可填写 `/wdl/data` 内已下载的本地模型路径；默认使用 Hub ID，自动下载到 `/wdl/data/huggingface/hub`。离线运行前先准备好权重、Processor 和数据缓存。

程序在导入 Hugging Face 库之前配置缓存目录，模型和数据加载还显式传入 `cache_dir`。`source scripts/server_env.sh` 同时配置 Conda、pip、torch 和临时目录，手动下载/安装前也应执行。代码不会迁移或删除已有缓存；旧数据需要另行复制或在新目录重新准备。

可通过环境变量 `OODEXPLORE_DATA_ROOT` 更换大文件根目录，或使用全局参数（放在子命令之前）：`python -m ood_explore --data-root /other/disk prepare`。显式 `--output` 仍优先控制产物位置；运行脚本的默认数据和输出路径随根目录变化。

默认 `device_map=auto`、BF16、SDPA。硬件不支持 BF16 时改为 `float16`；实际显存由图像分辨率、输出长度和分配方式决定。`max_pixels` 限制图像 token 开销；不能在 dev/test 之间更换设置。模型、数据的 `revision` 可换为固定 commit；运行清单会记录实际解析的模型 commit、依赖版本及评分代码 SHA256。

## 2. 服务器检查（不下载模型）

```bash
python -m pytest -q
```

包含冻结分位融合、错误正类方向、图像簇重采样、配对 token 对齐、答案解析和防止答案泄漏的协议检查；另用随机初始化的极小 GPT-2 验证实际 HF `generate` 与强制重放路径，无需权重下载。它不替代真实 Qwen 的多模态接入检查。

## 3. 准备固定数据

```bash
python -m ood_explore prepare --dev-count 100 --test-count 500 --seed 42
```

从 `derek-thomas/ScienceQA` 的官方 validation/test 分别抽取带图题。按 RGB 像素 hash 去重（开发集优先），不根据模型对错筛选。输出相对路径的 PNG、`dev.jsonl`、`test.jsonl`、来源与抽样清单。

模型只看 question、choices、image；不输入答案、lecture、solution、hint。答案仅留给解析后评价。每次准备须使用空目录，失败后请选新目录重试。手动构造相同 schema 的数据时，`image` 相对 JSONL 所在目录解析，`answer` 为零起始选项索引。

## 4. 十题真实 VLM 冒烟检查

```bash
python -m ood_explore run --config configs/qwen3b.json --data /wdl/data/oodexplore/datasets/scienceqa/dev.jsonl --output /wdl/data/oodexplore/runs/smoke --limit 10
```

默认前 10 题额外执行原图强制重放，比较相同位置的 NCI、fDBD、生成 token log 概率及负熵，误差超过容差即终止；每道题的检查结果存入 `verification`。这是新 VLM 适配检查，不重跑原论文实验。还应目视查看输出是否为正确的图片/选项、解释格式和完整答案。

生成和重放都通过原生 `generate` 管理多模态位置与 KV cache。hook 直接读取 **LM head 的输入与原始输出**，不是被 logits processor 修改过的分布。灰图通过自定义 processor 强制相同 token 序列；每次调用重新建立 cache。没有 `output_hidden_states=True` 或保存完整词表分布。

原图/灰图必须具有相同 `input_ids`、attention mask、视觉网格；不一致立即报错。仅存输出位置，EOS 不计分。灰图不独立作答，不给它分配原题错误标签。

## 5. 正式实验与续跑

```bash
python -m ood_explore run --config configs/qwen3b.json --data /wdl/data/oodexplore/datasets/scienceqa/dev.jsonl --output /wdl/data/oodexplore/runs/main/dev
python -m ood_explore run --config configs/qwen3b.json --data /wdl/data/oodexplore/datasets/scienceqa/test.jsonl --output /wdl/data/oodexplore/runs/main/test
python -m ood_explore analyze --dev /wdl/data/oodexplore/runs/main/dev/predictions.jsonl --test /wdl/data/oodexplore/runs/main/test/predictions.jsonl --output /wdl/data/oodexplore/runs/main/analysis
```

激活 Conda 环境后，也可执行 `bash scripts/run_server.sh`，自动依次完成准备、dev、test 和分析；默认数据目录 `/wdl/data/oodexplore/datasets/scienceqa`，输出目录 `/wdl/data/oodexplore/runs/main`，可通过脚本参数指定。该脚本使用当前环境的 Python，不自动切换环境、安装依赖或提交集群任务。

中断后用**相同命令追加 `--resume`**：

```bash
python -m ood_explore run --config configs/qwen3b.json --data /wdl/data/oodexplore/datasets/scienceqa/test.jsonl --output /wdl/data/oodexplore/runs/main/test --resume
```

每条完成后立即写入 JSONL；只接受原数据顺序的完整前缀。配置、代码、数据、图像、评分模块或依赖版本变化会拒绝续跑，避免混合实验。异常行不静默跳过。若进程恰在写入时被杀死导致最后一行损坏，需先备份并人工移除不完整的最后一行再续跑。请勿同时向同一输出目录启动两个进程。

smoke 与正式实验使用不同输出目录；不要将 `--limit 10` 的清单复用于正式全量运行。分析要求 dev/test 均完成，配置、实际模型版本和评分代码一致且无图像交集。

分析目录最后写入 `COMPLETE` 标记。若分析中断留下不完整目录，选新输出目录重新分析，或先备份不完整目录；脚本不会将它静默当成完成，也不会覆盖已有分析。

## 6. 分数和标签

对每个输出位置：`C = s(original)`，`G = s(original) - s(gray)`，分别使用 NCI、fDBD；整段 token 等权平均。每个分支保留其自身 argmax/top-k 的原评分定义。因此 G 描述预测几何整体变化，不保证比较的是同一词表类别。

开发集**有效输出、不使用正确性标签**建立经验 CDF：`F(x) = count(dev <= x) / n_dev`。联合风险：

```text
R_joint = ((1 - F_C(C)) + (1 - F_G(G))) / 2
```

开发分布冻结后用于测试；不搜索权重，不在测试集翻转分数方向。基线：NLL、entropy、NCI、fDBD、各自几何增益、同 token 概率增益，以及 NLL/概率增益的同样分位等权融合。主方法事先固定 `joint_fdbd`。

CSV/metrics 统一存储**风险方向**（越大越可能错误）。`logprob` 列实际为 `-mean_logprob`（NLL），`negative_entropy` 列为 `-negative_entropy`（熵），`*_gain` 列为负增益；名字沿用原始特征以便逐项核查。

解析规则为唯一一处 `Final answer: A`，选项须在有效范围内。主检测表仅用可解析、非截断、有评分的输出；全部样本准确率及解析失败/截断数另报。最终答案错误为正类，**不是每道错误均被标注为视觉幻觉**。图像不必要题、纯逻辑步骤也可能具有低 G。

## 7. 输出与解释

每个 run：

- `manifest.json`：模型实际版本、运行设置、数据/图像/源代码 hash、依赖和 GPU 信息。
- `predictions.jsonl`：答案、标签、原始双分支逐 token 分数、C/G、行号、验证结果、耗时与各 GPU 已分配显存峰值。

分析目录：

- `metrics.json` / `report.md`：AUROC、AP、错误比例、排除数量、1000 次成对图像簇 bootstrap，主方法相对 fDBD 与概率联合基线的差值区间、分阶段时间与显存。
- `reference.json` / `provenance.json`：冻结开发分布与来源。
- `scores.csv`：每题各方法的风险值。
- `quadrants.png` / `quadrants.json`：开发集中位数阈值划分的 C/G 四象限及错误比例。
- `manual_review.csv`：高 C/低 G 中固定随机抽取至多 20 条，供人工填写错误类型、视觉必要性和备注。
- `cases.md` / `cases/*.png`：按 ID 固定选取至多 5 正确、5 错误案例，显示原图、模型原文、分行 C/G 曲线。

AP 使用 sklearn average precision 定义；测试中只有单一类别时 AUROC 明确标为不可计算。少于 50 个有效错误会提示按预先方案扩展完整带图 test，而不按错误挑题。图像变化敏感性不等于正确视觉依据；teacher forcing 的文本历史可能已残留视觉信息。输出行只是过程观察单位，不能仅凭它宣称定位了真实内部错误起点。

所有图片引用保留数据目录，搬动分析目录时也需保持相对布局。各 GPU 峰值是每设备观察到的 allocated memory，不包含 allocator reserved memory，不将多卡峰值相加冒充同时总峰值。

## 8. 可选扩展

首轮完成后将配置复制为 `qwen7b.json`，把 `model` 改成 `Qwen/Qwen2.5-VL-7B-Instruct`，使用相同 data 文件、独立 dev/test/analysis 目录完整运行。不要把 3B 的开发分布用于 7B。

若错误样本太少，准备 `--test-count 0` 的新数据目录（相同 seed 和 dev-count），得到完整唯一图像测试集；重新运行该测试集，并和相同配置开发集分析。500 题结果与扩展结果分别报告。首轮不增加连续 latent 模型、层选择、漂移特征或监督 probe。

参考：[Qwen2.5-VL 官方文档](https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl)、[生成接口](https://huggingface.co/docs/transformers/en/internal/generation_utils)、[ScienceQA](https://github.com/lupantech/ScienceQA)。依赖版本固定为项目环境，不依赖滚动文档中的最新 API。
