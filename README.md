# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀）后，可以从最近成功的检查点继续训练；在**同环境、同种子、同
配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
scripts/verify_lr_probe.py # 端到端验证：POST /train/lr-probe 全部场景
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态/最近检查点续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| POST | `/train/lr-probe` | **学习率试算**：从最近成功检查点分别用两个候选 lr 走一步，返回更新前后该批 eval 交叉熵 |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史） |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt

# 学习率试算：两个不同、有限、正的学习率
curl -X POST localhost:8000/train/lr-probe -H 'Content-Type: application/json' \
     -d '{"lr_a": 0.005, "lr_b": 0.02}' | jq
```

## 学习率试算 `POST /train/lr-probe`

训练工程师在决定是否调整学习率前，用它比较两个候选对**同一批样本**的即时影响：

```json
{"lr_a": 0.005, "lr_b": 0.02}
```

响应（`lr_a`、`lr_b` 须为两个不同的、有限的正数，`0`/负数/`NaN`/`Infinity`/
字符串/布尔/多余字段均返回 422）：

示例响应（真实运行，`epochs=2, batch_size=32` 的默认配置）：

```json
{
  "run_id": "d22b96ea4482",
  "checkpoint": {"epoch": 0, "next_batch": 11, "completed": false,
                 "saved_at": "2026-11-18T08:13:47.374153+00:00"},
  "epoch": 0,
  "batch": 11,
  "step": 12,
  "batch_size": 32,
  "eval_loss_before": 0.4808414876461029,
  "candidates": [
    {"lr": 0.005, "eval_loss_before": 0.4808414876461029,
     "eval_loss_after": 0.4625001847743988, "delta": -0.0183413028717041},
    {"lr": 0.02,  "eval_loss_before": 0.4808414876461029,
     "eval_loss_after": 0.4112740159034729, "delta": -0.06956747174263}
  ]
}
```

`checkpoint` 是检查点原始位置；顶层 `epoch`/`batch` 是**被试算的那一批**，
`step` 是从 1 开始的全局批次序号。轮末检查点的例子：检查点
`{"epoch": 0, "next_batch": 19}`（默认配置 19 批/轮，位于第 0 轮末）→ 顶层
`"epoch": 1, "batch": 0, "step": 20`。

语义：

- **来源**：只用最近一次**成功保存**的检查点（原子写，所以一定是完整的）恢复
  模型、Adam 状态（含 step 计数）与 torch RNG；`run_id`/`epoch`/`batch` 均取自
  该检查点。
- **目标批**：检查点位置 `(epoch, next_batch)` 指向的下一批，样本排列与训练
  完全一致（`epoch_permutation(seed, epoch)`）。**若检查点位于轮末**
  （`next_batch == batches_per_epoch`），取**下一轮的首批**（响应的顶层
  `epoch`/`batch` 与之对应，`checkpoint` 保留检查点原始位置）。
- **两次试算共享起点**：各自从检查点完整复制模型和 Adam 状态，使用同一个由检查
  点 RNG 派生的私有 dropout 掩码，各做恰好一次 `backward + Adam.step`。因此两个
  候选的 `eval_loss_before` 必然相同；`lr_a` 的试算不会影响 `lr_b`。
- **前后损失**：更新前后都把模型切到 `eval()`（dropout 关闭），在**该批**上计算
  交叉熵；`delta = eval_loss_after - eval_loss_before`。
- **拒绝**：没有成功检查点 → 404；检查点已在最后一批之后（无后续批次）→ 409；
  无法确认检查点对应的原训练数据 → 409（见下）；试算发散出非有限损失 → 422。
- **不改变在线作业**：试算全在私有副本和私有 `torch.Generator` 上进行，不读写
  在线模型、优化器、全局 RNG、逐批损失，也不写检查点文件。混入任意多次试算后，
  后续逐批损失与最终参数仍与不试算的运行**逐位一致**（见验证脚本 P3/P4）。
- **`force=true`**：旧检查点立即从磁盘删除。已返回给客户端的结果仍带着旧
  `run_id`（响应是一次性快照，不随后续运行变化）；此后新的试算只使用新作业成功
  保存的检查点——若新作业还没写出第一个点，返回 404，绝不回退到已丢弃的旧点。
  试算在独立线程中执行，训练与 `GET /train/status`、`/train/losses` 继续响应。
- **数据可验证性**：每个检查点记录 `dataset.fingerprint`（特征列名 + X/y 字节 +
  形状的 SHA-256）。若检查点**未记录**指纹（例如旧版检查点），或当前加载的数据
  集指纹与之不符，试算一律拒绝（409）——绝不拿"不确定是不是同一批"的数据试算。

> 注意：试算用的是**已持久化**的检查点，运行中可能落后内存最多
> `checkpoint_every` 批；响应的 `checkpoint` 字段让这一点始终可见。

状态机：`idle → running ⇄ pausing → paused → running … → finished`（异常为
`failed`）。`paused` 表示训练已停、可续训；`checkpoint_dirty=true` 表示内存进度
领先于磁盘（上次保存失败），此时重启进程会回退到磁盘上的旧检查点。

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置与 `completed` 标志——即"准备执行第 k
   批"的完整现场。恢复 = 还原现场后继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs。
B–E、G 的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。

学习率试算有独立脚本：

```bash
python3 scripts/verify_lr_probe.py
```

真实拉起服务，验证 P1 参数校验（非有限/零/负/非数字/相等 lr、多余字段 → 422）；
P2 两候选共享起点、返回检查点 run_id/epoch/batch、重复试算确定；P3 暂停期间试算
不改损失历史、续训后与不中断基线逐位一致；P4 训练进行中并发试算不下扰、完成后
逐位一致；P5 轮末检查点取下一轮首批；P6 已无后续批次 → 409；P7 `force` 后旧结果
仍带旧 run_id、新请求只认新作业的点（尚无点 → 404，不回退）、状态查询持续响应；
P8 无指纹/异数据集检查点 → 409；P9 `/docs` 与 OpenAPI；P10 进程内证明「候选 lr
= 检查点 lr」时试算结果与真实训练一步**完全相同**，且试算后在线模型、Adam、
全局 RNG、损失历史均未变；P11 `force` 与进行中的试算竞争时，旧结果仍带旧
`run_id`，之后的试算只认新作业。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

（以下为 `python3 scripts/verify_repro.py` 在本仓库环境下的真实输出摘要）

```
    workdir: /tmp/verify-repro-ndor_el1
    
    == H: service basics (idle server) ==
      [PASS] H1 GET /
      [PASS] H2 GET /docs (Swagger UI)
      [PASS] H3 parameters before any checkpoint -> 404
      [PASS] H4 resume with no run -> 404
      [PASS] H5 pause while idle -> 409
    
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches, final loss 0.027771
    
    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted - pause latency 0.03s, stopped at batch 14
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] B3 pause+resume == uninterrupted
    
    == D: concurrent resume starts exactly one trainer ==
      [PASS] D1 concurrent resumes -> one 200, one 409 - got [200, 409]
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] D2 resumed run == uninterrupted
    
    == E1: checkpoint save failure keeps old checkpoint, retry works ==
      [PASS] E1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous checkpoint is intact on 
      [PASS] E1 status shows dirty checkpoint + error
      [PASS] E1 old checkpoint untouched
      [PASS] E1 retried pause -> 200, checkpoint persisted
      [PASS] E1 resume after retry
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E1 run after save-failure == uninterrupted
    
    == F: slow checkpoint saves do not block queries ==
      [PASS] F1 queries respond fast while a 1s save is in flight - worst status latency 1ms
      [PASS] F2 pause during slow saves
      [PASS] F3 slow-save run finishes
    
    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C1 server crashed mid-run
      [PASS] C2 restart recovers paused state from checkpoint - recovered at batch 10/95
      [PASS] C3 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] C4 kill+restart == uninterrupted
    
    == E2: save failure, then hard crash -> old checkpoint used ==
      [PASS] E2 pause with failing save -> 500
      [PASS] E2 restart falls back to old complete checkpoint - recovered at batch 6/95
      [PASS] E2 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E2 save-fail+kill == uninterrupted
    
    == G: hard crash in the middle of a checkpoint save ==
      (baseline: identical config, uninterrupted)
      [PASS] G1 server crashed during save
      [PASS] G2 no torn checkpoint / stale temp cleaned - leftovers: none
      [PASS] G3 restart recovers a complete checkpoint (old or new) - recovered at batch 1/19
      [PASS] G3b resume after restart
        losses: 19 entries, exact match
        params: all tensors bit-identical (sha256 1353a7415c7cae90 == 1353a7415c7cae90)
      [PASS] G4 kill-mid-save == uninterrupted
    
    ================ SUMMARY ================
      33/33 checks passed（逐项明细同上，全部 PASS）

```

`python3 scripts/verify_lr_probe.py`（同环境实测）：

```
    == P10: probe math matches a real training step; live state untouched ==
      [PASS] P10 probe leaves online model parameters untouched
      [PASS] P10 probe leaves the online Adam state untouched
      [PASS] P10 probe leaves the global torch RNG stream untouched
      [PASS] P10 probe leaves the loss history untouched
      [PASS] P10 candidate at the checkpoint lr equals one real training step - probe 0.7738370299339294 vs true 0.7738370299339294
      [PASS] P10 probe targeted the true next batch

    ================ SUMMARY ================
      [PASS] P3 resumed run == uninterrupted (losses / params)
      [PASS] P4 live-probed run == uninterrupted (losses / params)
      [PASS] P5 probe at an epoch-boundary checkpoint targets the next epoch's first batch
      [PASS] P6 probe on a finished run -> 409
      [PASS] P7 probe before the new run saved a point -> 404 (no fallback to old)
      [PASS] P7 probe after the new run has a checkpoint reports only the new run_id
      [PASS] P8 legacy checkpoint without fingerprint -> 409
      [PASS] P8 checkpoint from a different dataset -> 409
      [PASS] P9 /train/lr-probe documented in OpenAPI
      [PASS] P11 in-flight probe still reports the old run_id
      [PASS] P11 post-force probe reports only the new run_id
      52/52 checks passed（上述为摘要，完整 52 项全部 PASS）
```

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 检查点只保留最新一份（原子替换保证新旧二选一）；未做历史版本保留/滚动清理。
- `force=true` 启动新运行会**立即丢弃**旧检查点：若新运行在写出第一个检查点
  前崩溃，则无点可恢复（状态回到 `idle`），只能重新开始——这是刻意语义，
  避免"已丢弃的旧运行"在重启后被悄悄恢复。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。
- 学习率试算基于**已持久化**检查点，运行中最多落后内存 `checkpoint_every` 批；
  且只做**一步** Adam 更新的即时比较，不是多步/整轮的学习率搜索。
- 试算要求检查点带数据集指纹。指纹只覆盖特征列名、X/y 内容与形状，**不含文件
  路径**：数据内容相同但换路径仍可试算（内容不同则拒绝）。升级前写入的旧检查点
  没有指纹，升级后会被拒绝试算（但续训不受影响），重新跑一次新运行即可。
- 试算串行执行（一把进程内锁）：并发多个试算会排队，单次计算量很小；若某个
  lr 使损失变为 NaN/Inf，该试算整体返回 422 而不返回部分结果。
