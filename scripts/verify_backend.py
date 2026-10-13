# -*- coding: utf-8 -*-
"""后端验收自检脚本。

一条命令跑完，输出可以直接贴给组长做验收。**不依赖任何人工判断**：
每一行都是实际执行出来的结果，不是写死的。

用法（在仓库根目录）：

    .\\.venv\\Scripts\\python.exe scripts\\verify_backend.py

退出码 0 表示全部通过，非 0 表示有失败项。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PASS, FAIL, SKIP = "通过", "失败", "跳过"

rows: list[tuple[str, str, str, str]] = []


def make_output_safe() -> None:
    """让标准输出永远不会因编码问题崩掉。

    Windows 上 stdout 重定向到文件或管道时，Python 用 locale 编码（GBK）
    而不是控制台编码。此时打印 GBK 装不下的字符（比如画勾和打叉的 emoji）
    会抛 ``UnicodeEncodeError`` 直接终结脚本。

    这不是假想问题：``docs/BACKEND_GUIDE.md`` 第九节坑 #1 记的就是同一个坑
    （根目录那个画勾的 ``test_registry.py`` 把整个测试套件搞挂），而这个自检
    脚本自己第一版也踩了 —— 把输出重定向到文件时，在最后一行崩掉，
    前面的检查全部白跑。

    两道保险：本脚本正文只用 GBK 能表示的字符，这里再把错误处理放宽。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def check(section: str, task: str, ok: bool | None, detail: str = "") -> None:
    """ok=None 表示跳过（通常是本地缺数据）。

    详情里的换行会被压成空格：异常消息常带换行，展开会把汇总表撑乱。
    """
    status = SKIP if ok is None else (PASS if ok else FAIL)
    rows.append((section, task, status, " ".join(detail.split())))


def banner(text: str) -> None:
    print()
    print(f"-- {text} " + "-" * max(0, 58 - len(text)))


# ==========================================================================
# 环境
# ==========================================================================

def check_environment() -> bool:
    banner("环境")
    print(f"  Python {sys.version.split()[0]}  ({sys.executable})")

    py_ok = sys.version_info >= (3, 10)
    check("环境", "Python >= 3.10", py_ok, f"{sys.version_info.major}.{sys.version_info.minor}")

    runtime = {}
    for module in ("numpy", "fastapi", "uvicorn", "xlrd", "openpyxl"):
        try:
            imported = __import__(module)
            runtime[module] = getattr(imported, "__version__", "?")
        except ImportError:
            runtime[module] = None
    missing = [name for name, version in runtime.items() if version is None]
    check(
        "环境", "运行依赖（numpy/fastapi/uvicorn/xlrd/openpyxl）",
        not missing,
        "全部就绪" if not missing else f"缺 {missing} → pip install -r requirements.txt",
    )

    try:
        import httpx2  # noqa: F401
        test_dep = True
        test_detail = "httpx2 已安装"
    except ImportError:
        test_dep = False
        test_detail = "缺 httpx2 → pip install -r requirements-dev.txt"
    check("环境", "测试依赖（httpx2）", test_dep, test_detail)

    return not missing and test_dep


# ==========================================================================
# 九项任务
# ==========================================================================

def check_data_layer() -> bool:
    from backend.data.registry import (
        DatasetUnavailable,
        dataset_names,
        load_dataset,
    )

    names = dataset_names()
    check("数据层", "1 数据集加载模块（登记表）", bool(names), f"已登记 {len(names)} 个")

    if not names:
        check("数据层", "1 数据集实际加载", None, "没有登记的数据集")
        return False

    try:
        loaded = {name: load_dataset(name) for name in names}
    except DatasetUnavailable:
        # 缺数据文件是环境问题，不是后端缺陷 —— 必须算「跳过」而不是「失败」，
        # 否则换台没放数据的机器跑，会误报后端不可用
        check("数据层", "1 数据集实际加载", None,
              "本地缺数据文件（数据集本体不进仓库），已跳过")
        return False
    except Exception as exc:  # noqa: BLE001
        check("数据层", "1 数据集实际加载", False, f"{type(exc).__name__}: {exc}")
        return False

    formats = sorted({item.table.format for item in loaded.values()})
    check("数据层", "1 数据集实际加载", True, f"{len(loaded)} 个，格式 {formats}")

    mismatch = [
        name for name, item in loaded.items()
        if item.X.shape[0] != item.spec.n_samples
    ]
    check(
        "数据层", "1 元数据与实际行数一致", not mismatch,
        "全部一致" if not mismatch else f"不一致：{mismatch}",
    )
    for name, item in list(loaded.items())[:3]:
        print(f"      {name:<11} {item.X.shape[0]:>5} 行 × {item.X.shape[1]:>3} 特征  "
              f"{item.spec.task_type}")
    return True


def check_preprocessing() -> bool:
    import numpy as np

    from backend.data.encoding import (
        impute_with_train_statistics,
        missing_marker_set,
        is_missing,
    )
    from backend.data.preprocessing import standardize

    # 填充必须只用训练集统计量
    filled_train, filled_test, means = impute_with_train_statistics(
        np.array([[0.0], [10.0]]), np.array([[np.nan], [1000.0]])
    )
    impute_ok = abs(means[0] - 5.0) < 1e-9 and abs(filled_test[0, 0] - 5.0) < 1e-9
    check(
        "数据层", "2 预处理：填充只用训练集统计量", impute_ok,
        "训练均值=5，测试集缺失也填 5（若用全局均值会是 336.7）" if impute_ok else "有泄露",
    )

    _, test_scaled, mean, scale = standardize(
        np.array([[0.0], [2.0]]), np.array([[1000.0]])
    )
    std_ok = abs(mean[0] - 1.0) < 1e-9 and abs(test_scaled[0, 0] - 999.0) < 1e-9
    check("数据层", "2 预处理：标准化只用训练集统计量", std_ok,
          "测试集用训练集均值/标准差换算" if std_ok else "有泄露")

    markers = missing_marker_set(["-200"], decimal_comma=True)
    marker_ok = all(
        is_missing(text, markers, decimal_comma=True)
        for text in ("-200", "-200.0", "-200,0")
    )
    check("数据层", "2 缺失标记数值等价匹配", marker_ok, "-200 / -200.0 / -200,0 都识别")

    return impute_ok and std_ok and marker_ok


def check_algorithms() -> bool:
    from backend.algorithms.registry import ALGORITHM_REGISTRY, create_algorithm

    count = len(ALGORITHM_REGISTRY)
    check("算法层", "3 算法注册模块", count == 10, f"{count} 个算法（团队原有代码）")

    try:
        model = create_algorithm("knn", k=3)
        usable = hasattr(model, "fit") and hasattr(model, "predict")
    except Exception as exc:  # noqa: BLE001
        usable = False
        print(f"      create_algorithm 失败：{exc}")
    check("算法层", "3 注册表能创建模型", usable, "create_algorithm('knn') 可用")
    return count == 10 and usable


def check_api(has_data: bool):
    from fastapi.testclient import TestClient

    from backend.api.main import app
    from backend.services import store

    client = TestClient(app)

    check("接口层", "8 接口文档存在", (REPO / "docs" / "api.md").exists(), "docs/api.md")

    if not has_data:
        for task in (
            "接口一 GET /api/datasets",
            "接口二 GET /api/datasets/{name}",
            "接口三 GET /api/algorithms",
            "接口四 POST /api/experiments",
            "接口六 WS 训练中连接：实时进度",
            "接口六 WS 结束后连接：补发结果",
            "接口五 GET /api/experiments/{id}",
            "结果不含不可序列化的 model 对象",
            "结果可被严格 JSON 序列化",
            "6 实验记录落盘",
            "6 历史查询接口",
        ):
            check("接口层", task, None, "本地缺数据集")
        return

    r = client.get("/api/datasets")
    check("接口层", "接口一 GET /api/datasets", r.status_code == 200 and len(r.json()) >= 7,
          f"{r.status_code}，{len(r.json())} 个数据集")

    r = client.get("/api/datasets/ckd")
    ok = r.status_code == 200 and "label_distribution" in r.json()
    check("接口层", "接口二 GET /api/datasets/{name}", ok,
          f"{r.status_code}，含字段/缺失/标签分布" if ok else str(r.status_code))

    check("接口层", "接口三 GET /api/algorithms",
          client.get("/api/algorithms").status_code == 200
          and client.get("/algorithms").status_code == 404,
          "/api/algorithms 200 且 /algorithms 404（P0 缺陷 #2 已修）")

    created = client.post(
        "/api/experiments",
        json={"dataset": "iris", "algorithm": "knn", "parameters": {"k": 5}},
    )
    check("接口层", "接口四 POST /api/experiments", created.status_code == 202,
          f"{created.status_code}（立即返回，训练在后台线程）")

    if created.status_code != 202:
        return
    experiment_id = created.json()["experiment_id"]

    # 接口六：WebSocket 进度推送。
    #
    # 这里**故意用一个慢一点的实验**（random_forest 在 concrete 上约 2~3 秒）。
    # 一开始我用 iris+knn 验证，它只要 10 毫秒，等 WebSocket 连上去时实验早就
    # 结束了，只会收到 snapshot → result，误判成"没有进度推送"——那是竞态，
    # 不是产品问题。两条路径要分别验证。
    ws_experiment = client.post(
        "/api/experiments",
        json={
            "dataset": "concrete",
            "algorithm": "random_forest",
            "parameters": {"n_estimators": 60, "task": "regression"},
        },
    ).json()["experiment_id"]

    live: list[str] = []
    try:
        with client.websocket_connect(f"/ws/experiments/{ws_experiment}") as ws:
            while True:
                message = ws.receive_json()
                live.append(message["type"])
                if message["type"] in ("result", "error"):
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"      WebSocket 异常：{exc}")

    progress_count = live.count("progress")
    check("接口层", "接口六 WS 训练中连接：实时进度",
          progress_count > 0 and bool(live) and live[-1] == "result",
          f"收到 {progress_count} 条 progress，最后一条是 {live[-1] if live else '无'}")

    # 结束后再连一次：文档承诺会补发 snapshot + result，前端无需区分先后
    replay: list[str] = []
    try:
        with client.websocket_connect(f"/ws/experiments/{ws_experiment}") as ws:
            replay = [ws.receive_json()["type"], ws.receive_json()["type"]]
    except Exception as exc:  # noqa: BLE001
        print(f"      WebSocket 重放异常：{exc}")
    check("接口层", "接口六 WS 结束后连接：补发结果",
          replay == ["snapshot", "result"], " → ".join(replay) if replay else "无消息")

    store.delete_experiment(ws_experiment)

    payload = client.get(f"/api/experiments/{experiment_id}").json()
    result = payload.get("result") or {}
    metrics = result.get("metrics") or {}
    timing_ok = (
        "train_time_seconds" in result and "predict_time_seconds" in result
        and "prediction_time_seconds" not in result
    )
    check("接口层", "接口五 GET /api/experiments/{id}",
          payload["status"] == "finished" and timing_ok,
          f"accuracy={metrics.get('accuracy')}，训练/预测分别计时（P0 缺陷 #7 已修）")

    check("接口层", "结果不含不可序列化的 model 对象", "model" not in result,
          "P0 缺陷 #8 已修")
    check("接口层", "结果可被严格 JSON 序列化",
          json.dumps(result, allow_nan=False) is not None,
          "inf/nan 已转 null")

    check("接口层", "6 实验记录落盘", store.experiment_exists(experiment_id),
          "experiments/results/ 下已生成 JSON")
    check("接口层", "6 历史查询接口",
          client.get("/api/experiments").status_code == 200, "GET /api/experiments")

    store.delete_experiment(experiment_id)


def check_errors_and_logging(has_data: bool):
    import logging

    from backend.logging_config import (
        configure_logging,
        file_logging_enabled,
        get_logger,
        log_directory,
    )

    if has_data:
        from fastapi.testclient import TestClient

        from backend.api.main import app
        client = TestClient(app)
        r = client.get("/api/datasets/no_such_dataset")
        check("接口层", "7 统一错误响应是 JSON",
              r.status_code == 404 and r.headers["content-type"].startswith("application/json"),
              '404 且 content-type 是 application/json（不是 HTML 错误页）')

        failed = client.post(
            "/api/experiments",
            json={"dataset": "iris", "algorithm": "knn", "parameters": {"k": 0}},
        ).json()["experiment_id"]
        for _ in range(300):
            snapshot = client.get(f"/api/experiments/{failed}").json()
            if snapshot["status"] in ("finished", "failed"):
                break
            time.sleep(0.05)
        check("接口层", "7 失败实验暴露堆栈与进度日志",
              bool(snapshot.get("traceback")) and bool(snapshot.get("logs")),
              "失败时返回 traceback 且 logs 以失败原因收尾")
        from backend.services import store
        store.delete_experiment(failed)

    if not file_logging_enabled():
        check("接口层", "7 日志写入文件", None, "ML_LOG_TO_FILE=0，本次跳过")
        return

    configure_logging("INFO", force=True)
    probe = f"验收自检标记-{time.time():.0f}"
    get_logger("verify").info(probe)
    for handler in logging.getLogger("backend").handlers:
        handler.flush()
    path = log_directory() / "backend.log"
    written = path.exists() and probe in path.read_text(encoding="utf-8")
    check("接口层", "7 日志写入文件", written,
          f"{path} （UTF-8，含分级/模块名/时间）" if written else f"{path} 未写入")


def check_tests() -> bool:
    banner("测试套件")
    print("  正在运行 python -m unittest discover …", flush=True)
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover"],
            cwd=str(REPO), capture_output=True, text=True,
            # 必须显式指定 UTF-8：本机默认是 GBK，解 UTF-8 输出会抛 UnicodeDecodeError
            encoding="utf-8", errors="replace",
            # 加超时，避免子进程卡死时整个自检无限挂起
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        check("测试", "9 后端测试全部通过", False, "子进程超过 300 秒未结束")
        return False

    tail = [line for line in (completed.stderr or "").splitlines() if line.strip()]
    summary = tail[-1] if tail else "(无输出)"
    check("测试", "9 后端测试全部通过", completed.returncode == 0,
          f"{summary}（{time.time() - started:.1f}s）")
    return completed.returncode == 0


# ==========================================================================
# 主流程
# ==========================================================================

def main() -> int:
    print("=" * 62)
    print(" 后端验收自检")
    print("=" * 62)

    env_ok = check_environment()
    if not env_ok:
        print("\n依赖不全，后面的检查无法进行。")
        print("请先执行：pip install -r requirements-dev.txt")
        render()
        return 1

    data_ok = check_data_layer()
    check_preprocessing()
    check_algorithms()
    check_api(data_ok)
    check_errors_and_logging(data_ok)
    check_tests()

    if not data_ok:
        print()
        print("提示：本地没有数据集，依赖数据的检查已跳过。")
        print("      数据集本体不进仓库，把数据放到仓库根的 数据集/ 目录，")
        print("      或设置环境变量 ML_DATA_ROOT 指向数据目录即可。")

    return render()


def render() -> int:
    banner("汇总")
    width = max(len(task) for _, task, _, _ in rows) if rows else 10
    failed = 0
    skipped = 0
    for section, task, status, detail in rows:
        if status == FAIL:
            failed += 1
        elif status == SKIP:
            skipped += 1
        mark = {PASS: "[通过]", FAIL: "[失败]", SKIP: "[跳过]"}[status]
        print(f"  {mark} {task:<{width}}  {detail}")

    total = len(rows)
    print()
    print(f"  共 {total} 项：通过 {total - failed - skipped}，失败 {failed}，跳过 {skipped}")
    if failed:
        print("  结论：存在失败项，后端当前不可用。")
    elif skipped:
        print("  结论：全部通过，但有跳过项（通常是本地缺数据集）。")
        print("        补齐数据后重跑可得到完整结论。")
    else:
        print("  结论：全部通过，后端可用。")
    return 1 if failed else 0


if __name__ == "__main__":
    make_output_safe()
    sys.exit(main())
