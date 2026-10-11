# Porto Medium v0.2 学生使用指南

本交付包用于复现 PWC 风格的在线 PWE 选择实验、分析候选模型的互补性，并开展后续算法研究。下载处理好的数据即可运行，不需要私有城市数据、GPU、OSM 在线服务或原始 GPS 文件。

首先完成数据校验和快速复现，再运行完整实验。当前版本固定为连续 168 小时、每小时 5 条测试订单。这里的 TravelTime 和 Preference 是公开轨迹构造的代理权重，不能当作原会议版 ConSTGAT、POWER-S、POWER-D 模型的完整复现。

## 1. 获取代码与运行环境

使用 Python 3.11 或 3.12。交付验证覆盖 macOS ARM64 上的 Python 3.12 和 Linux 上参考实验使用的 Python 3.11。建议至少 4 GB 内存、2 GB 可用磁盘空间；完整实验使用 CPU。

可以直接下载 Release 中的 `porto-medium-v0.2-student-kit.tar.gz`，解压后进入 `PWC-student-kit/`。整包已经包含代码、数据、环境锁定文件和指南，此时跳过下面的 Git 获取命令与第 2 节的数据下载，只需建立环境并校验数据。依赖安装完成后可以离线运行；整包不包含 Python 本体或依赖安装包。

```bash
git clone https://github.com/GaoYucen/PWC.git
cd PWC
git checkout porto-medium-v0.2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-porto-eval.lock.txt
python -m pytest -q
```

若安装的是 Python 3.11，将 `python3.12` 改为 `python3.11`。Windows 的环境激活命令为 `.venv\Scripts\activate`；原始数据构建脚本使用 `fork`，建议在 Linux 或 macOS 上运行。处理好数据的 Windows 实验入口未在本次交付中验证。

`requirements-porto-eval.lock.txt` 是处理后数据的实验环境；`requirements-porto-build.lock.txt` 保存原构建环境的精确包版本，只有重建地图匹配和候选权重时才需要。不要用通用 `requirements.txt` 代替这个版本的锁定环境。

## 2. 下载并校验数据

```bash
python scripts/public_benchmark/download_porto_medium.py
python scripts/public_benchmark/student_benchmark.py verify
```

下载器从固定版本的 GitHub Release 获取数据，验证压缩包与清单 SHA-256，再解压到 `data/porto-medium-hourly-v0.2/`。文件校验通过后才读取订单 pickle。不要加载来源不明的 pickle 文件。

手动下载时也可使用同一下载器完成校验与解压：

```bash
python scripts/public_benchmark/download_porto_medium.py \
  --archive /path/to/porto-medium-hourly-v0.2.tar.gz
```

下载器不覆盖已有数据目录。如果目录已经存在，直接运行 `verify`；校验不通过时换一个新目录下载。外部数据路径可通过 `--data /path/to/porto-medium-hourly-v0.2` 传给学生入口。

## 3. 数据范围与文件说明

时间使用 `Europe/Lisbon`，按行程开始时间划分，区间左闭右开。

| 划分 | 本地时间区间 | 匹配成功轨迹数 |
| --- | --- | ---: |
| Train | 2014-04-07 至 2014-04-28 | 9,697 |
| Validation | 2014-04-28 至 2014-05-05 | 3,178 |
| Test | 2014-05-05 至 2014-05-12 | 2,956 |

路网包含 56,779 个节点和 130,561 条有向边。Test 的 168 个小时全部至少有 6 条匹配轨迹；正式实验每小时取按开始时间排序的前 5 条，共 840 条订单。Validation 每小时使用全部匹配轨迹，最少 2 条、中位数 19 条，按小时等权平均。

| 文件 | 用途 |
| --- | --- |
| `graph.graphml` | 构建时使用的冻结 OSM 路网，供原始数据重建 |
| `nodes.csv.gz`, `edges.csv.gz` | 原 OSM 节点编号与有向边索引，包含坐标、长度、道路属性 |
| `topology.npz` | 路由所需的稠密节点索引和 CSR 拓扑 |
| `trajectories_train/val/test.jsonl.gz` | 匹配后的轨迹，含时间、OSM 节点路径、边路径和匹配质量信息 |
| `sampled_orders.pkl` | Test 的 168 个小时订单列表；节点路径使用稠密索引 |
| `validation_orders.pkl` | Validation 的 168 个小时订单列表 |
| `weight_length.npy` | 物理道路长度 |
| `weight_time.npy` | 训练轨迹估计的静态时间成本 |
| `weight_power_s.npy` | 训练轨迹估计的静态路线偏好代理成本 |
| `weight_power_d.npy` | 168 个 hour-of-week 的动态偏好代理成本 |
| `weights/` | 原构建器的完整权重输出，还含动态时间成本和训练边访问次数 |
| `reference/reward_matrices.npz` | Validation 和 Test 的 `168 × 4` 基线奖励缓存 |
| `reference/summary.json` | 原 20-seed 完整实验结果 |
| `metadata.json`, `PROTOCOL.json` | 轨迹覆盖、过滤统计和固定实验口径 |
| `MANIFEST.json` | 文件大小、SHA-256、源数据和源代码版本 |
| `DATA_ATTRIBUTION.md` | 数据来源、署名和再分发条件 |

`topology.npz` 的 `node_ids` 将稠密索引映射回 OSM ID；权重数组与边表的 `edge_idx` 对齐。静态成本形状为 `(130561,)`，动态成本形状为 `(168, 130561)`，值越小越适合最短路选择。静态和动态偏好成本由训练轨迹的道路使用频次构造，并经过收缩；它们不是心理偏好真值。

## 4. 先做快速复现

```bash
python scripts/public_benchmark/student_benchmark.py replay \
  --output results/porto-replay.json

python scripts/public_benchmark/student_benchmark.py smoke \
  --hours 4 --output results/porto-smoke.json
```

`replay` 重新执行验证集选参、20-seed direct selector 评估和 switching oracle，检查结果与参考记录一致。它使用已保存的奖励矩阵，不重新计算路由或 chasing。

`smoke` 重新计算前 4 小时的基线路由，并执行一个 FPL/chasing 序列。基线奖励与缓存比较的绝对容差为 `1e-7`；检查通过会输出 `max_baseline_reward_error`。4 小时的 overlap 不应与 168 小时的结果直接比较。

## 5. 完整复现 168 小时实验

```bash
python scripts/public_benchmark/hourly_tracking_benchmark.py \
  --city-dir data/porto-medium-hourly-v0.2 \
  --output results/porto-medium-v02-N5-seeds20 \
  --N 5 --seeds 20
```

这条命令重新计算 Validation/Test 的基线路由、在 Validation 上选择参数，并重新计算 20 个随机种子的 Test chasing 路由。原服务器实验约 13 分钟，个人电脑耗时取决于 CPU。先用 `--seeds 1` 检查完整运行入口也可以，但单 seed 的结果不能代替下表。

输出包括 `summary.json`、`reward_matrices.npz`、`per_seed.json` 和 `provenance.json`。每次使用新的输出目录，保留自己的配置、结果与代码版本。

实验结束后自动对照参考结果：

```bash
python scripts/public_benchmark/student_benchmark.py compare \
  --results results/porto-medium-v02-N5-seeds20 \
  --output results/porto-medium-v02-comparison.json
```

| 方法 | Direct sampled mean | Chasing mean |
| --- | ---: | ---: |
| FPL beta 3 | 61.255791% | 61.237659% |
| FPL validation tuned | 61.490411% | 61.456467% |
| Hedge | 61.439118% | 61.393501% |
| Fixed Share | 61.110970% | 61.197568% |
| Discounted Hedge | 61.292875% | 61.166330% |
| Sliding Hedge | 61.100837% | 61.022012% |

Best fixed 是 Preference-Static，61.576742%；direct full oracle 是 64.432814%。Direct sampled mean 与概率选择器的 expected direct reward 是不同的统计量，应使用相同列进行比较。

原结果保存在 `reports/porto-medium-v0.2/reference_summary.json`。同一环境下可对照到至少 `1e-6` 个百分点；如果差异超出范围，先检查锁定环境、文件校验、参数与随机种子，不要通过调整参数来追平参考数字。

## 6. 固定实验语义

- 所有候选权重只使用 Train 轨迹构造。选择器参数只在 Validation 上选择。
- Test 开始时选择器从零历史重新启动；没有把 Validation 的累计奖励作为初始化。
- 在小时 `t` 做选择时，只使用 `t-1` 及以前的奖励；当前小时的四个基线奖励在选择后作为 full-information 反馈更新。
- 奖励是共同有向道路的物理长度占实际匹配路径长度的比例，重复经过的边按次数计入。先对小时内订单取均值，再对 168 小时取均值。
- 跨 PWE 先做正数 median 校准；chasing 保留前 3 个小时与当前小时，系数为 `exp(-delta)`。
- 动态基线自身也使用相同的前 3 小时加当前小时衰减。动态时间成本虽然随数据发布，但本次四专家实验没有纳入它。
- FPL 测试 seeds 为 0–19；其他概率选择器采样 seeds 为 1000–1019。20 seeds 共用同一周和同一份订单。
- `beta=3` 是会议重建的冻结参考参数；Porto 验证集选择的 FPL 参数是 `eta=100`。不能将二者混为一个结果。

## 7. 可以开展的研究与结论边界

可以先复现现有算法、分析时段和 OD 特征下的模型互补性、检查地图匹配质量，再设计新选择器。当前所有在线方法仍低于 Test 上的最佳固定模型，应保留这些负结果。

这是一周、一个订单样本的 benchmark。20 seeds 反映算法随机性，不是跨周泛化。Test 已被查看并用于探索时间平滑窗口；后续新方法可以用这周做开发诊断，正式验收需要新的未查看时段。不要将已有的测试集选窗口结果当作未污染的测试证据。

地图匹配是 nearest-node snap 加有界物理最短路连接，属于保守代理标签，不能替代独立匹配精度验证。冻结 OSM 图是 2026 年获取的路网，并非 2014 年历史快照。按行程开始时间分组，没有截断跨划分边界的行程，也没有模拟真实行程结束后的延迟反馈。需要研究延迟反馈时另建协议。

N=10 只有 153/168 个 Test 小时满足要求。不要丢弃空缺小时或重复订单后仍声称保持本版连续周协议。改变 matcher、抽样方式、专家集合或反馈时序，应建立新数据版本并保留 v0.2。

Direct oracle 只比较四个基线的直接路由；混合历史权重可能产生新的路径，因此它不是完整 chasing 系统的严格上界。切换次数曲线也只描述事后空间，不能证明最优 PWE 的变化可以被提前预测。

## 8. 从原始公开数据重建

默认使用已发布的冻结数据，不需要执行这一节。需要研究数据处理时，在 Linux 或 macOS 上建立单独环境：

```bash
python3.12 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install -r requirements-porto-build.lock.txt
mkdir -p rebuild-porto
cp data/porto-medium-hourly-v0.2/graph.graphml rebuild-porto/porto_drive.graphml
python scripts/public_benchmark/porto_hourly_week.py \
  --root rebuild-porto --raw-zip /path/to/train.csv.zip \
  --test-week-start 2014-05-05 --train-weeks 3 --val-weeks 1 \
  --raw-cap-per-hour 200 --workers 8 --seed 20261011 \
  --name porto-medium-hourly-v0.2
```

原始数据由 [UCI 官方数据页](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015) 获取。内部 `train.csv.zip` 的 SHA-256 为 `210dd0a20da66a8fc2de3440aecd84670921bc257591f8365a4475e31453c5ea`。UCI 外层下载压缩包的哈希不等于这个内部文件的哈希。

使用包内冻结 GraphML；重新下载实时 OSM 图会改变路网和匹配结果。构建器不自动生成本交付的 Release 清单与参考实验结果。重建后应先对照轨迹数、每小时覆盖和权重，而不是假定 gzip 文件字节必然相同。

## 9. 数据来源与署名

数据由 UCI Porto taxi trajectories 与 OpenStreetMap 构造，分别保留 CC BY 4.0 来源署名和 ODbL 条件。具体说明见 [数据署名与再使用说明](PORTO_DATA_ATTRIBUTION.md)。原始私有城市数据不在本交付包内。

使用研究方法时引用仓库 README 中的 KDD 2024 论文；使用公开 benchmark 时同时引用 UCI 数据 DOI，并注明 OpenStreetMap contributors 和 Porto Medium v0.2 版本。
