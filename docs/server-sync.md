# 服务器部署与 Git 同步

仓库：`https://github.com/ZERMORE/OODExplore.git`，分支 `main`。服务器代码目录 `/home/wxh/projects/OODexplore`；权重、数据、环境、缓存、实验结果都放在 `/home/wxh/wdl/data`。Linux 路径区分大小写。

## 1. 先在本地提交并推送本次修改

以下为本地 Windows PowerShell 命令，不是在服务器上执行：

```powershell
Set-Location E:\Research\LLMCoT\methods\OODExplore
git status
git diff
git add README.md .gitignore .gitattributes configs/qwen3b.json ood_explore/backend.py ood_explore/cli.py ood_explore/data.py ood_explore/storage.py scripts/run_server.sh scripts/server_env.sh tests/test_storage.py docs/server-sync.md
git diff --cached --stat
git commit -m "Route large files to /home/wxh/wdl/data and document server sync"
git push origin main
```

确认暂存列表只含本次预期修改后再提交。这里提供操作指引；助手没有替你执行 commit 或 push。如果已提交这些修改，直接 push 即可。Git 仅同步代码及配置，模型/数据/实验结果不放入仓库。

## 2. 服务器首次克隆

```bash
mkdir -p /home/wxh/projects /home/wxh/wdl/data
git clone --branch main https://github.com/ZERMORE/OODExplore.git /home/wxh/projects/OODexplore
cd /home/wxh/projects/OODexplore
git log -1 --oneline
```

`/home/wxh/projects/OODexplore` 必须不存在或为空；如果已经是该仓库，请使用后面的更新步骤，不重复 clone。目录须由当前账户有写权限。

私有仓库须先配置服务器的 GitHub SSH key，或通过 Git 凭据管理器认证；不要把 token 写进 clone URL、配置文件或脚本。使用 SSH 时对应 URL 为 `git@github.com:ZERMORE/OODExplore.git`。

已有复现代码也必须在服务器可读，默认位置 `/home/wxh/projects/OODmethod/ood_repro/scoring.py`。该文件来自你已完成的复现仓库，不包含在 OODExplore 中；请同步现有复现仓库或把配置中的 `oodmethod_dir` 指向其实际位置，无需再次跑原论文实验。

## 3. 在数据盘创建 Conda 环境

若你已经创建了 `(oodexplore)` 环境，直接继续使用 `conda activate oodexplore`，跳过创建环境的两条命令即可；数据路径配置与 Conda 环境位置无关。

```bash
cd /home/wxh/projects/OODexplore
source scripts/server_env.sh
conda env create --prefix /home/wxh/wdl/data/conda/envs/oodexplore -f environment.yml
conda activate /home/wxh/wdl/data/conda/envs/oodexplore
```

然后按 README 的 CUDA 版本选择安装 PyTorch；以下为 CUDA 12.4 构建示例，适用前提见 README：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
python -m pip check
python -m ood_explore show-config --config configs/qwen3b.json
```

Conda 包缓存、pip 缓存和临时目录由环境脚本路由到数据盘；程序自动将 Hugging Face 权重和数据缓存路由到数据盘。`pip install -e .` 让后续 Git 更新的 Python 源码直接生效，不用每次重装。

可先按 README 执行服务器测试和十题检查，再启动正式实验：

```bash
bash scripts/run_server.sh
```

结果默认在 `/home/wxh/wdl/data/oodexplore/runs/main`，不会随 Git 更新被覆盖。

## 4. 日常同步：本地 push → 服务器 pull

如果终端曾保留旧版根目录变量，更新后执行 `export OODEXPLORE_DATA_ROOT="$HOME/wdl/data"`，再 source 环境脚本。新终端未设置变量时将自动采用该目录。

本地修改并提交后 `git push origin main`。服务器在没有实验正在运行时执行：

```bash
cd /home/wxh/projects/OODexplore
git status --short
git pull --ff-only origin main
git log -1 --oneline
source scripts/server_env.sh
conda activate /home/wxh/wdl/data/conda/envs/oodexplore
```

拉取前确认没有未处理的服务器代码修改。`--ff-only` 避免自动生成合并提交；若提示分支分叉或本地修改冲突，先保存并处理这些修改，不使用 `reset --hard` 强制覆盖。[Git pull 官方说明](https://git-scm.com/docs/git-pull)

如果本次更新了 `pyproject.toml` 中的依赖，重新执行：

```bash
python -m pip install -e .
python -m pip check
```

如果修改的是 `environment.yml` 中的 Conda 基础依赖：

```bash
conda env update --prefix /home/wxh/wdl/data/conda/envs/oodexplore -f environment.yml
```

不要在正在运行的实验期间 pull 到同一代码目录。若必须同时保留多个版本，使用不同 checkout/工作目录，并使用不同输出目录。

## 5. 续跑与新版本实验

仅在代码、配置和依赖均不变时续跑：

```bash
python -m ood_explore run --config configs/qwen3b.json --data /home/wxh/wdl/data/oodexplore/datasets/scienceqa/test.jsonl --output /home/wxh/wdl/data/oodexplore/runs/main/test --resume
```

程序会校验代码和评分模块 hash。pull 后代码变化，旧结果不能混入新版本；请选择新实验输出目录：

```bash
bash scripts/run_server.sh configs/qwen3b.json /home/wxh/wdl/data/oodexplore/datasets/scienceqa /home/wxh/wdl/data/oodexplore/runs/run-002
```

这会复用已下载的模型缓存和已整理的数据，不必重复下载。新版本的 dev/test 应配套重跑，避免跨版本合并结果。

若要继续旧实验，可以先完成/保存当前工作后切回其原 commit，并恢复原依赖；模型/数据盘文件继续保留。原模型版本也需与运行清单一致。

## 6. 常用位置

| 内容 | 默认位置 |
|---|---|
| 项目代码 | `/home/wxh/projects/OODexplore` |
| 原复现评分器 | `/home/wxh/projects/OODmethod/ood_repro/scoring.py` |
| Hugging Face 权重与原始文件 | `/home/wxh/wdl/data/huggingface/hub` |
| datasets Arrow 缓存 | `/home/wxh/wdl/data/huggingface/datasets` |
| Xet 下载缓存 | `/home/wxh/wdl/data/huggingface/xet` |
| 固定 ScienceQA 子集 | `/home/wxh/wdl/data/oodexplore/datasets/scienceqa` |
| 实验输出 | `/home/wxh/wdl/data/oodexplore/runs` |
| Conda 环境 | `/home/wxh/wdl/data/conda/envs/oodexplore` |

缓存变量须在导入库之前设置；CLI 和环境脚本均遵循这一顺序。[Hugging Face 环境变量说明](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)

路径调整不会自动搬运既有权重或数据，也不会删除旧缓存。若 `/home/wxh/wdl/data` 为多人共享盘，使用自己有权限的子目录，并通过 `OODEXPLORE_DATA_ROOT` 配置；认证凭据勿放在对所有人可读的位置。
