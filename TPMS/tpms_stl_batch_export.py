"""
tpms_stl_batch_export.py
------------------
TPMS 批量导出 STL 脚本 —— 基于 tpms_batch_run.py 的 ntopcl 调用机制改写。

跟 tpms_batch_run.py 的核心区别:
  tpms_batch_run.py : 单个notebook, 循环不同density, 每次输出一个FEA结果csv
  本脚本            : 循环文件夹里的多个notebook(每个对应一种TPMS类型),
                       每个notebook内部再循环该类型CSV里筛选出的多个density,
                       每次输出一个STL(Mesh from Implicit Body -> Export Mesh)

保留 -> ntopcl调用机制(-v2 -j in.json -o out.json notebook.ntop)、
        实时日志流+心跳、按run分文件夹存log/json、DIY density过滤逻辑
        (DENSITY_RANGE/DENSITY_LIST/DENSITY_STEP，逻辑原样照搬)。
去掉 -> SAVE_NTOP机制(这里每个notebook要循环很多个density,没必要每次
        都把结果写回notebook本身)、重试逻辑(按你的要求,失败直接跳过记日志)。
新增 -> 按notebook文件名(不含后缀)去CSV_FOLDER下找同名csv、
        Tolerance作为脚本顶层统一DIY变量(不像t_max/t_min那样逐行变化)、
        STL文件校验(validate_stl，逻辑思路跟validate_csv一致但检查内容不同)。

工作流:
  1) python tpms_stl_batch_export.py check      # 检查路径、notebook<->csv匹配情况
  2) python tpms_stl_batch_export.py run        # 按CONFIG批量跑
     python tpms_stl_batch_export.py run --dry-run   # 先看一遍计划，不真的调用ntopcl
     python tpms_stl_batch_export.py run --force     # 已存在的stl也重新跑一遍
     python tpms_stl_batch_export.py run --max 5     # 最多跑5个run就停(跨notebook累计)

notebook里需要预先设置好(不需要脚本处理):
  - Lattice Type 下拉选择(每个notebook各自对应哪个类型,已经选好)
  - Mesh: Mesh from Implicit Body (Body已连接到implicit body, Sharpen/Simplify已设置好)
  - Export Mesh 的 Mesh 输入已连接到 Mesh from Implicit Body 的输出
本脚本只通过ntopcl写入 t_max / t_min / Tolerance / Path 四个顶层input。
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ============================================================
# CONFIG —— 三个DIY路径 + Tolerance + density范围，按需修改
# ============================================================

# nTop Automate 可执行文件（跟tpms_batch_run.py保持一致）
NTOPCL = r"C:\Program Files\nTopology\nTopology\ntopcl.exe"

# --- DIY 1: ntop和csv读取文件夹路径 ---
NTOP_FOLDER = Path(r"D:\project\week7\111")   # 存放多个 {Type}.ntop 的文件夹
CSV_FOLDER = Path(r"D:\project\week7\csv")     # 存放多个 {Type}.csv 的文件夹（跟ntop同名）

# --- DIY 3: 保存stl的路径 ---
OUTPUT_FOLDER = Path(r"D:\project\week7\stl model")   # 输出根目录，脚本会在下面按类型建子文件夹

# --- Tolerance: 所有模型统一用这一个值，手动改这里即可 ---
TOLERANCE = 0.2   # mm

# notebook 里对应的 input 名字（要跟notebook里显示的名字完全一致，大小写/空格都要对上）
NB_TMAX = "t_max"
NB_TMIN = "t_min"
NB_TOLERANCE = "Tolerance"
NB_PATH = "Path"          # Export Mesh 的 Path 输入

# --- DIY 2: density的Step Size和Range ---
# 两者都给的话 DENSITY_LIST 优先；两者都是 None 表示跑查表CSV里的全部行
DENSITY_RANGE = (0.09, 0.31)       # 例如 (0.10, 0.20) 只跑 density 落在这个闭区间内的行
DENSITY_LIST = None        # 例如 [0.10, 0.20, 0.30] 只跑这几个指定值（按最接近的一行匹配）
DENSITY_STEP = 0.02        # 例如 0.02；None = 不做步长抽稀（会跟DENSITY_RANGE叠加使用）
DENSITY_STEP_START = None  # None = 以(筛完区间后)csv里最小的density为起点对齐

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_ROOT = SCRIPT_DIR / "runs_stl"                    # 每次运行的log/json都存这里


# ============================================================
# 基础工具（跟tpms_batch_run.py完全一致，原样照搬）
# ============================================================
def die(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)


def stream_process(args, log_path=None, cwd=None):
    """跑一个命令，实时打印stdout，附带耗时和心跳。"""
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
            if time.time() - last >= 30:
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
def build_input_json(t_max, t_min, tolerance, out_stl: Path, dest_json: Path):
    """只写 t_max/t_min/Tolerance/Path 四个字段，其余input用notebook默认值。

    t_max/t_min: 无单位level set标量(scalar，不带units)。
    Tolerance: 带mm单位的长度量，跟tpms_batch_run.py里Edge length同样写法
    (type仍是"scalar"，额外带"units":"mm"，ntopcl不认"quantity"这个类型名)。
    Path: file_path类型，直接是完整的输出stl文件路径(含文件名)。"""
    data = {
        "inputs": [
            {"name": NB_TMAX, "type": "scalar", "values": t_max},
            {"name": NB_TMIN, "type": "scalar", "values": t_min},
            {"name": NB_TOLERANCE, "type": "scalar", "units": "mm", "values": tolerance},
            {"name": NB_PATH, "type": "file_path",
             "values": str(out_stl).replace("\\", "/")},
        ]
    }
    dest_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {NB_TMAX: t_max, NB_TMIN: t_min, NB_TOLERANCE: f"{tolerance} mm",
            NB_PATH: str(out_stl)}


# ============================================================
# 查表 CSV 读取 + DIY 范围过滤（过滤逻辑跟tpms_batch_run.py完全一致）
# ============================================================
def load_lookup_rows(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
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
            if abs(n - round(n)) < 1e-6:
                picked.append(r)
        rows = picked

    return rows


# ============================================================
# 结果校验：ntopcl有没有真的产出一个可用的STL
# ============================================================
def validate_stl(path: Path) -> tuple[bool, str]:
    """只管"这次ntopcl有没有正常产出一个像样的STL文件"，不解析具体三角面片内容。

    检查内容:
      1) 文件存在
      2) 文件大小明显大于一个空/半成品文件的量级
         (binary STL header固定84字节起步，真实geometry的mesh通常远大于这个数，
          用几KB做下限，能筛掉"写了个头就中断"的半成品文件)
      3) 尝试读取文件头，binary STL开头84字节是 80字节描述+4字节三角形数量(uint32)，
         如果这个数量字段读出来是0，说明是个"技术上合法但没有任何三角面片"的空壳STL
    """
    if not path.exists():
        return False, "stl不存在"

    size = path.stat().st_size
    if size <= 2000:
        return False, f"stl太小({size}字节)，很可能是空文件/写了一半"

    try:
        with open(path, "rb") as f:
            header = f.read(84)
            if len(header) < 84:
                return False, "文件头不足84字节，不是完整的binary STL"
            # ascii STL开头是"solid "，binary STL开头80字节任意但第81-84字节是三角形数量
            if header[:5].lower() == b"solid":
                # ascii STL，不做三角形数量解析，只要size够大就认为OK
                return True, "OK (ascii)"
            import struct
            tri_count = struct.unpack("<I", header[80:84])[0]
            if tri_count == 0:
                return False, "binary STL三角形数量为0，空壳文件"
            return True, f"OK (binary, {tri_count} triangles)"
    except Exception as e:
        return False, f"stl打不开: {e}"


# ============================================================
# 单次运行（对应一个 ntop文件 x 一个density）
# ============================================================
def run_one(ntop_path: Path, type_name: str, density: float, t: float,
            force: bool, dry_run: bool) -> tuple[bool, float, str]:
    t_max = t
    t_min = -t

    out_dir = OUTPUT_FOLDER / type_name
    out_stl = out_dir / f"{type_name}_{density:.2f}.stl"

    stem = f"{type_name}_d{density:.2f}"
    run_dir = RUNS_ROOT / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    in_json = run_dir / "in.json"
    out_json = run_dir / "out.json"
    log_txt = run_dir / "log.txt"

    if out_stl.exists() and not force:
        ok, reason = validate_stl(out_stl)
        print(f"[skip] {out_stl.name} 已存在，校验结果: {reason} (用 --force 重跑)")
        return ok, 0.0, reason

    out_dir.mkdir(parents=True, exist_ok=True)
    fed = build_input_json(t_max, t_min, TOLERANCE, out_stl, in_json)

    print(f"\n{'=' * 60}")
    print(f"  RUN {stem}")
    for k, v in fed.items():
        print(f"    {k} = {v}")
    print(f"    STL -> {out_stl}")
    print(f"{'=' * 60}")

    if dry_run:
        print("  [dry-run] 不实际调用 ntopcl")
        return True, 0.0, "dry-run"

    # 不加 -s：这里每个notebook要循环跑很多个density，没必要每次都把
    # mesh结果写回notebook本身（会持续覆盖，且没有下游用途）。
    args = [NTOPCL, "-v2", "-j", str(in_json), "-o", str(out_json), str(ntop_path)]
    rc, secs = stream_process(args, log_path=log_txt)

    if rc != 0:
        reason = f"ntopcl exit code {rc}（看log: {log_txt}）"
        print(f"  exit={rc}  time={secs:.0f}s  {reason}")
        return False, secs, reason

    ok, reason = validate_stl(out_stl)
    print(f"  exit={rc}  time={secs:.0f}s  校验: {reason}")
    return ok, secs, reason


# ============================================================
# 命令
# ============================================================
def discover_pairs():
    """遍历NTOP_FOLDER，找出每个notebook对应的csv(同名)，返回配对列表。
    csv不存在的notebook会被记录到missing列表，check/run都能看到。"""
    if not NTOP_FOLDER.exists():
        die(f"NTOP_FOLDER不存在: {NTOP_FOLDER}")

    ntop_files = sorted(NTOP_FOLDER.glob("*.ntop"))
    pairs = []
    missing = []
    for ntop_path in ntop_files:
        type_name = ntop_path.stem
        csv_path = CSV_FOLDER / f"{type_name}.csv"
        if csv_path.exists():
            pairs.append((ntop_path, type_name, csv_path))
        else:
            missing.append((type_name, csv_path))
    return pairs, missing


def cmd_check():
    print(f"[check] ntopcl:       {NTOPCL}  {'OK' if os.path.exists(NTOPCL) else 'MISSING'}")
    print(f"[check] NTOP_FOLDER:  {NTOP_FOLDER}  {'OK' if NTOP_FOLDER.exists() else 'MISSING'}")
    print(f"[check] CSV_FOLDER:   {CSV_FOLDER}  {'OK' if CSV_FOLDER.exists() else 'MISSING'}")
    print(f"[check] OUTPUT_FOLDER:{OUTPUT_FOLDER}")
    print(f"[check] TOLERANCE:    {TOLERANCE} mm")
    print(f"[check] 将写入notebook的input名字: "
          f"t_max={NB_TMAX!r}  t_min={NB_TMIN!r}  "
          f"tolerance={NB_TOLERANCE!r}  path={NB_PATH!r}")
    print("  （这几个名字要跟notebook里显示的名字完全一致，大小写/空格都算）")

    pairs, missing = discover_pairs()
    print(f"\n[check] 找到 {len(pairs)} 个 notebook<->csv 配对成功:")
    for ntop_path, type_name, csv_path in pairs:
        print(f"    {type_name}: {ntop_path.name}  <->  {csv_path.name}")

    if missing:
        print(f"\n[warn] {len(missing)} 个notebook没找到对应csv，将被跳过:")
        for type_name, csv_path in missing:
            print(f"    {type_name}: 期望路径 {csv_path} 不存在")


def cmd_run(args):
    pairs, missing = discover_pairs()
    if missing:
        print(f"[warn] {len(missing)} 个notebook没找到对应csv，将跳过: "
              f"{[m[0] for m in missing]}")
    if not pairs:
        die("没有任何notebook<->csv配对成功，检查NTOP_FOLDER/CSV_FOLDER设置")

    # 先把整体计划(跨所有notebook)列出来，方便dry-run时一眼看清楚
    plan = []
    for ntop_path, type_name, csv_path in pairs:
        rows = load_lookup_rows(csv_path)
        rows = apply_diy_filter(rows)
        if not rows:
            print(f"[warn] {type_name} 过滤后没有任何density可跑，跳过")
            continue
        for row in rows:
            plan.append((ntop_path, type_name, row["density"], row["t"]))

    if not plan:
        die("过滤后没有任何run可跑，检查 DENSITY_RANGE / DENSITY_LIST / DENSITY_STEP 设置")

    print(f"[plan] 即将跑 {len(plan)} 个run，覆盖 {len(pairs)} 个类型")

    results = []
    for i, (ntop_path, type_name, density, t) in enumerate(plan):
        if args.max_runs is not None and i >= args.max_runs:
            print(f"[stop] --max {args.max_runs} 到达，剩余的下次再跑")
            break
        ok, secs, reason = run_one(ntop_path, type_name, density, t,
                                    force=args.force, dry_run=args.dry_run)
        results.append((type_name, density, ok, secs, reason))

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for type_name, density, ok, secs, reason in results:
        mark = "OK  " if ok else "FAIL"
        tail = "" if ok else f"  <- {reason}"
        print(f"  {type_name:<12} density={density:<6} {mark}  {secs:.0f}s{tail}")

    n_fail = sum(1 for _, _, ok, _, _ in results if not ok)
    if n_fail:
        print(f"\n[warn] {n_fail}/{len(results)} 个run没通过校验，"
              f"上面SUMMARY里 '<- 原因' 就是具体问题，对应log在 {RUNS_ROOT}\\<type>_d<density>\\log.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("check", help="检查路径配置、notebook<->csv匹配情况")

    p_run = sub.add_parser("run", help="按CONFIG批量跑")
    p_run.add_argument("--force", action="store_true", help="已存在的stl也重跑")
    p_run.add_argument("--dry-run", action="store_true", help="只打印计划，不调用ntopcl")
    p_run.add_argument("--max", type=int, dest="max_runs", help="最多跑N个run就停（跨notebook累计）")

    args = ap.parse_args()
    if args.cmd == "check":
        cmd_check()
    elif args.cmd == "run":
        cmd_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
