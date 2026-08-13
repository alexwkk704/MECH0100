"""
tpms_batch_run.py
------------------
TPMS (Slotted-P / C(+-Y) / IWP ...) 批量 nTop homogenization 跑批脚本。


  保留 -> ntopcl 调用机制 (ntopcl -v2 -j in.json -o out.json notebook.ntop)、
          实时日志流 + 心跳、按 run 分文件夹存 log/json、
          -s 保存算完结果的notebook副本（同学原脚本里的save_ntop机制）。
  去掉 -> Seed自动重试、Inner Size/Size Multi命名、xlsx模板+resume指纹、
          Notebook/Tag列覆盖、ntopcl -t schema解析 —— 前几项是同学ADMS
          notebook专属的用不到，最后一项是因为choice类型的input会让
          ntopcl -t直接报错退出。
  新增 -> 直接从 density-t 查表CSV读取自变量，自动算 t_max/t_min/edge_length；
          DENSITY_RANGE / DENSITY_LIST / DENSITY_STEP 三个DIY开关，方便
          小范围测试脚本。

工作流:
  1) python tpms_batch_run.py check      # 检查路径都对不对
  2) python tpms_batch_run.py run        # 按 CONFIG 里的密度范围批量跑
     python tpms_batch_run.py run --dry-run   # 先看一遍计划，不真的调用ntopcl
     python tpms_batch_run.py run --force     # 已存在的csv也重新跑一遍

不再调用 `ntopcl -t` 生成完整schema，直接手写只含 t_max/t_min/Edge length/Path
四个字段的最小JSON传给ntopcl，没写到的input一律用notebook里的默认值。
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# ============================================================
# CONFIG —— 每次换 notebook / geometry 时检查这一整块
# ============================================================

# nTop Automate 可执行文件
NTOPCL = r"D:\Program Files\nTopology\nTopology\ntopcl.exe"

# 当前要跑的 notebook（例如 IWP Generator）
NTOP_FILE = r"D:\Program Files\dzz_project\script\D' Generator.ntop"

# density -> t 查表CSV（格式: density,t,vf_actual）
# 每个geometry一份，换notebook时手动改这里指向对应的表
LOOKUP_CSV = r"D:\Program Files\dzz_project\script\D' Generator.csv"

# edge_length = EDGE_C * t，EDGE_C是你自己定的常数，换模型手动改
EDGE_C = 2

# notebook 里对应的 input 名字（要跟notebook里显示的名字完全一致，
# 大小写/空格都要对上，不对上的话会被 build_input_json 报错提醒）
NB_TMAX = "t_max"
NB_TMIN = "t_min"
NB_EDGE = "Edge length"
NB_PATH = "Path"          # Section 2 Export Material 的那个 Path

# 最终 homogenize 结果 csv 存放的文件夹
OUTPUT_DIR = Path(r"D:\Program Files\dzz_project\script\matrix")

# ---- 保存算完FEA结果/mesh的.ntop文件（能在nTop里直接打开看结果，不用重新算） ----
# 原理: ntopcl加 -s 这个flag会把mesh+结果直接写回notebook文件本身，
# 所以要先复制一份notebook再对副本操作，不会动你的主notebook文件。
SAVE_NTOP = True
NTOP_SAVE_DIR = OUTPUT_DIR.parent / "ntop_saved"   # 汇总存放，文件名跟csv同名

# ---- DIY 测试范围：只想跑一部分density时改这两个开关 ----
# 两者都给的话 DENSITY_LIST 优先；两者都是 None 表示跑查表CSV里的全部行
DENSITY_RANGE = (0.10, 0.30)      # 例如 (0.10, 0.20) 只跑 density 落在这个闭区间内的行
DENSITY_LIST = None       # 例如 [0.10, 0.20, 0.30] 只跑这几个指定值（按最接近的一行匹配）

# ---- DIY 步长抽稀：不改查表CSV本身，只是从表里按这个间隔挑行 ----
# 例如查表CSV本身是0.01一行(0.05,0.06,0.07...)，这里设0.02就只挑 0.05,0.07,0.09...
# 挑选起点默认是查表CSV里最小的density；如果想指定别的起点，改 DENSITY_STEP_START
# 会跟 DENSITY_RANGE 叠加使用(先按区间筛，再按步长抽稀)；DENSITY_LIST给了就直接忽略这两个
DENSITY_STEP = 0.02        # 例如 0.02；None = 不做步长抽稀
DENSITY_STEP_START = None  # None = 以(筛完区间后)查表CSV里最小的density为起点对齐

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = SCRIPT_DIR / "runs"                    # 每次运行的log/json都存这里


# ============================================================
# 基础工具
# ============================================================
def die(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)


def stream_process(args, log_path=None, cwd=None):
    """跑一个命令，实时打印stdout，附带耗时和心跳（原脚本机制原样保留）。"""
    t0 = time.time()
    logf = open(log_path, "a", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=cwd,
        encoding="utf-8", errors="replace",
    )
    stop_hb = threading.Event()

    def hb():
        last = time.time()
        while not stop_hb.is_set():
            time.sleep(2)
            if time.time() - last >= 300:
                print(f"  [heartbeat] {int(time.time() - t0)}s elapsed", flush=True)
                last = time.time()

    threading.Thread(target=hb, daemon=True).start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [{int(time.time() - t0):>4}s] {line}", flush=True)
                if logf:
                    logf.write(line + "\n")
    except KeyboardInterrupt:
        print("\n[ABORT] Ctrl+C - terminating ntopcl...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stop_hb.set()
        if logf:
            logf.close()
        sys.exit(130)
    proc.wait()
    stop_hb.set()
    if logf:
        logf.close()
    return proc.returncode, time.time() - t0


# ============================================================
# 手写最小 input JSON
# ============================================================
def build_input_json(t_max, t_min, edge_length, out_csv: Path, dest_path: Path):
    """只写 t_max/t_min/Edge length/Path 四个字段，其余input用notebook默认值。

    注意：t_max/t_min 是无单位的level set标量(scalar，不带units字段)，但
    Edge length 在notebook里是带mm单位的长度量。ntopcl不认"quantity"这个类型名
    (会报 unrecognized type)，正确做法是type仍然写"scalar"，只是额外带上
    "units":"mm" 这个字段——纯写scalar不带units会报
    "requires units of length, but received no units"。"""
    data = {
        "inputs": [
            {"name": NB_TMAX, "type": "scalar", "values": t_max},
            {"name": NB_TMIN, "type": "scalar", "values": t_min},
            {"name": NB_EDGE, "type": "scalar", "units": "mm", "values": edge_length},
            {"name": NB_PATH, "type": "file_path",
             "values": str(out_csv).replace("\\", "/")},
        ]
    }
    Path(dest_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {NB_TMAX: t_max, NB_TMIN: t_min, NB_EDGE: f"{edge_length} mm",
            NB_PATH: str(out_csv)}


# ============================================================
# 查表 CSV 读取 + DIY 范围过滤
# ============================================================
def load_lookup_rows():
    p = Path(LOOKUP_CSV)
    if not p.exists():
        die(f"查表CSV不存在: {p}")
    rows = []
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "density": float(r["density"]),
                "t": float(r["t"]),
            })
    rows.sort(key=lambda x: x["density"])
    return rows


def apply_diy_filter(rows):
    if DENSITY_LIST is not None:
        picked = []
        for target in DENSITY_LIST:
            closest = min(rows, key=lambda r: abs(r["density"] - target))
            picked.append(closest)
        return picked

    if DENSITY_RANGE is not None:
        lo, hi = DENSITY_RANGE
        rows = [r for r in rows if lo <= r["density"] <= hi]

    if DENSITY_STEP is not None and rows:
        start = DENSITY_STEP_START if DENSITY_STEP_START is not None else rows[0]["density"]
        picked = []
        for r in rows:
            n = (r["density"] - start) / DENSITY_STEP
            if abs(n - round(n)) < 1e-6:   # 容差判断是不是落在step的整数倍上
                picked.append(r)
        rows = picked

    return rows


# ============================================================
# 结果校验：ntopcl有没有真的跑成功
# ============================================================
def validate_csv(path: Path) -> tuple[bool, str]:
    """检查输出csv是不是一个完整、可用的文件。
    只管"这次ntopcl有没有正常产出"，不判断数值本身合不合理（比如对称性/正定性）。

    注意：nTop导出的矩阵csv每行末尾习惯多打一个逗号（除了最后一行），
    这会在每行末尾产生一个多余的空列，属于正常导出格式，不是数据缺失，
    所以先把每行末尾的空单元格去掉，再做长度/异常值检查。"""
    if not path.exists():
        return False, "csv不存在"

    size = path.stat().st_size
    if size <= 100:
        return False, f"csv太小({size}字节)，很可能是空文件/写了一半"

    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            raw_rows = list(csv.reader(f))
    except UnicodeDecodeError:
        return False, "csv编码异常，读取报错（很可能没写完整）"
    except Exception as e:
        return False, f"csv打不开: {e}"

    # 去掉每行末尾的空单元格（trailing comma导致的），只保留真正有内容的部分
    rows = []
    for row in raw_rows:
        while row and row[-1].strip() == "":
            row = row[:-1]
        if row:  # 跳过整行都是空的（比如文件末尾的空行）
            rows.append(row)

    if len(rows) < 1:
        return False, "csv去掉空行/空列后没有任何数据"

    row_len = len(rows[0])
    for i, row in enumerate(rows, start=1):
        if len(row) != row_len:
            return False, f"第{i}行列数({len(row)})跟第1行({row_len})对不上，文件可能没写完整"
        for cell in row:
            c = cell.strip().lower()
            if c in ("nan", "inf", "-inf", "infinity", "-infinity", ""):
                return False, f"第{i}行出现异常值 ({cell!r})"

    return True, "OK"


# ============================================================
# 单次运行
# ============================================================
def run_one(density: float, t: float, force: bool, dry_run: bool) -> tuple[bool, float, str]:
    t_max = t
    t_min = -t
    edge_length = EDGE_C * t

    stem = f"{Path(NTOP_FILE).stem}_d{density:.2f}"
    out_csv = OUTPUT_DIR / f"{stem}.csv"

    run_dir = RUNS_ROOT / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    in_json = run_dir / "in.json"
    out_json = run_dir / "out.json"
    log_txt = run_dir / "log.txt"

    if out_csv.exists() and not force:
        ok, reason = validate_csv(out_csv)
        print(f"[skip] {out_csv.name} 已存在，校验结果: {reason} (用 --force 重跑)")
        return ok, 0.0, reason

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fed = build_input_json(t_max, t_min, edge_length, out_csv, in_json)

    print(f"\n{'=' * 60}")
    print(f"  RUN {stem}")
    for k, v in fed.items():
        print(f"    {k} = {v}")
    print(f"    CSV -> {out_csv}")
    print(f"{'=' * 60}")

    if dry_run:
        print("  [dry-run] 不实际调用 ntopcl")
        return True, 0.0, "dry-run"

    # SAVE_NTOP开着的话：先把notebook复制一份给这次运行专用，
    # ntopcl加-s会把mesh+结果写回这份副本，不会碰你的主notebook文件
    if SAVE_NTOP:
        ntop_target = str(run_dir / f"{stem}.ntop")
        shutil.copy(NTOP_FILE, ntop_target)
    else:
        ntop_target = NTOP_FILE

    args = [NTOPCL, "-v2"]
    if SAVE_NTOP:
        args.append("-s")
    args += ["-j", str(in_json), "-o", str(out_json), ntop_target]
    rc, secs = stream_process(args, log_path=log_txt)

    if rc != 0:
        reason = f"ntopcl exit code {rc}（看log: {log_txt}）"
        print(f"  exit={rc}  time={secs:.0f}s  {reason}")
        return False, secs, reason

    ok, reason = validate_csv(out_csv)
    print(f"  exit={rc}  time={secs:.0f}s  校验: {reason}")

    if ok and SAVE_NTOP:
        NTOP_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        final_ntop = NTOP_SAVE_DIR / f"{stem}.ntop"
        shutil.copy(ntop_target, final_ntop)
        print(f"  [saved] 带结果的notebook已保存 -> {final_ntop}")

    return ok, secs, reason


# ============================================================
# 命令
# ============================================================
def cmd_check():
    print(f"[check] ntopcl:     {NTOPCL}  {'OK' if os.path.exists(NTOPCL) else 'MISSING'}")
    print(f"[check] notebook:   {NTOP_FILE}  {'OK' if os.path.exists(NTOP_FILE) else 'MISSING'}")
    print(f"[check] lookup csv: {LOOKUP_CSV}  {'OK' if os.path.exists(LOOKUP_CSV) else 'MISSING'}")
    print(f"[check] output dir: {OUTPUT_DIR}")
    print(f"[check] EDGE_C: {EDGE_C}")
    print(f"[check] SAVE_NTOP: {SAVE_NTOP}"
          + (f"  -> 保存到 {NTOP_SAVE_DIR}" if SAVE_NTOP else "  (不保存算完的notebook)"))
    print(f"[check] 将写入notebook的input名字: "
          f"t_max={NB_TMAX!r}  t_min={NB_TMIN!r}  edge={NB_EDGE!r}  path={NB_PATH!r}")
    print("  （这几个名字要跟notebook里显示的名字完全一致，大小写/空格都算；"
          "如果第一次跑 run --dry-run 后CSV没生成，先怀疑这里拼错了）")


def cmd_run(args):
    rows = load_lookup_rows()
    rows = apply_diy_filter(rows)
    if not rows:
        die("过滤后没有任何density可跑，检查 DENSITY_RANGE / DENSITY_LIST / DENSITY_STEP 设置")

    print(f"[plan] 即将跑 {len(rows)} 个density: "
          f"{[r['density'] for r in rows]}")

    results = []
    for i, row in enumerate(rows):
        if args.max_runs is not None and i >= args.max_runs:
            print(f"[stop] --max {args.max_runs} 到达，剩余的下次再跑")
            break
        ok, secs, reason = run_one(row["density"], row["t"],
                                    force=args.force, dry_run=args.dry_run)
        results.append((row["density"], ok, secs, reason))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for density, ok, secs, reason in results:
        mark = "OK  " if ok else "FAIL"
        tail = "" if ok else f"  <- {reason}"
        print(f"  density={density:<6} {mark}  {secs:.0f}s{tail}")

    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    if n_fail:
        print(f"\n[warn] {n_fail}/{len(results)} 个run没通过校验，"
              f"上面SUMMARY里 '<- 原因' 就是具体问题，对应log在 {RUNS_ROOT}\\<stem>\\log.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("check", help="检查路径配置")

    p_run = sub.add_parser("run", help="按CONFIG批量跑")
    p_run.add_argument("--force", action="store_true", help="已存在的csv也重跑")
    p_run.add_argument("--dry-run", action="store_true", help="只打印计划，不调用ntopcl")
    p_run.add_argument("--max", type=int, dest="max_runs", help="最多跑N个就停")

    args = ap.parse_args()
    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "run":
        cmd_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
