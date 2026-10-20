"""一键 HTML 测试报告（设计文档 §3.8 报告层）。

自包含单文件报告：KPI 卡 + 图表 base64 内联 + 明细表，不依赖模板引擎
（字符串模板即可）。数据源全部为各阶段产物（parquet/CSV/JSON/PNG/STDF），
报告层不重新跑仿真。

口径说明：
- 良率双口径：测试限口径来自 CP 结果表（hard_bin），规格限口径来自 W1
  存档（lot_dies.csv.gz 的 pass 列）；
- 检出率/过杀率：testexec.cp.detection_report 事后 join 存档标签
  （与 fa/cluster.py 相同的验证动作，标签不进报告明细）；
- STDF 记录数：stdf.reader.count_stdf_records 只扫记录头。
"""

from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "outputs"


def _esc(s: object) -> str:
    return _html.escape(str(s), quote=True)


def _img_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def count_records(stdf_path: Path) -> int:
    from stdf.reader import count_stdf_records
    return count_stdf_records(stdf_path)


def shmoo_boundary(shmoo_summary: dict) -> str:
    """从 Shmoo summary JSON 提取 DEF_TIMING 失效边界（KPI 卡用）。"""
    try:
        corners = shmoo_summary["dies"]["DEF_TIMING"]["fail_corners_vt"]
        temps = sorted({float(t) for t, _ in corners})
        vmin = shmoo_summary.get("fixed", {}).get("vt_load_a", 2.0)
        return f"T≥{temps[0]:.0f}°C × ILOAD={vmin:g} A（VIN 全范围贯通）"
    except (KeyError, IndexError):
        return "见 Shmoo 图"


def collect_default_inputs(outputs_dir: Path = DEFAULT_OUT) -> dict:
    """从 outputs/ 收集报告输入（main.py / 独立运行用）。"""
    outputs_dir = Path(outputs_dir)
    with open(outputs_dir / "analysis" / "test_time_optimization.json",
              encoding="utf-8") as f:
        time_report = json.load(f)
    with open(outputs_dir / "fa" / "fa_report.json", encoding="utf-8") as f:
        fa_report = json.load(f)
    with open(outputs_dir / "char" / "shmoo_summary.json", encoding="utf-8") as f:
        shmoo_summary = json.load(f)
    cpk_table = pd.read_csv(outputs_dir / "analysis" / "cpk_table.csv")
    images = {
        "Wafer Map（25 片 fail 空间分布）": outputs_dir / "wafermap.png",
        "Soft Bin 帕累托": outputs_dir / "analysis" / "bin_pareto.png",
        "SPC I-MR 控制图（干净 lot vs 注入漂移）": outputs_dir / "analysis" / "spc_imr.png",
        "参数相关矩阵": outputs_dir / "analysis" / "correlation_matrix.png",
        "Shmoo V×T（@2A）": outputs_dir / "char" / "shmoo_vt.png",
        "Shmoo V×Load（@125°C）": outputs_dir / "char" / "shmoo_vl.png",
        "VOUT3 参数分布": outputs_dir / "char" / "dist_vout3.png",
    }
    return {
        "results": pd.read_parquet(outputs_dir / "cp_results.parquet"),
        "lot": pd.read_csv(outputs_dir / "lot_dies.csv.gz"),
        "cpk_table": cpk_table,
        "fa_report": fa_report,
        "shmoo_summary": shmoo_summary,
        "time_report": time_report,
        "stdf_path": outputs_dir / "lot_cp.stdf",
        "images": images,
    }


def generate_report(out_path: str | Path,
                    results: pd.DataFrame,
                    lot: pd.DataFrame,
                    cpk_table: pd.DataFrame,
                    fa_report: dict,
                    shmoo_summary: dict,
                    time_report: dict,
                    stdf_path: Path,
                    images: dict[str, Path],
                    lot_id: str = "LOT20261020A",
                    program: str = "PMU300_CP_v1.2",
                    gen_date: str | None = None) -> Path:
    """生成自包含 HTML 报告 → out_path。KPI 全部由输入数据现算。"""
    from testexec import cp as cp_mod
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    date = gen_date or _dt.date.today().isoformat()

    # CSV 读回时空标签为 NaN，归一化回空串（检出率统计按标签字面量匹配）
    lot = lot.copy()
    if "defect" in lot.columns:
        lot["defect"] = lot["defect"].fillna("").astype(str)

    # ---- KPI 计算 ----
    y_test = float((results["hard_bin"] == 1).mean())
    y_spec = float(lot["pass"].mean()) if "pass" in lot.columns else y_test
    det = cp_mod.detection_report(results, lot)
    stdf_mb = Path(stdf_path).stat().st_size / 1e6
    n_rec = count_records(stdf_path)
    cpk_min = cpk_table.sort_values("cpk").iloc[0]
    dppm = (1 - y_test) * 1e6
    hb = results["hard_bin"].value_counts()
    hb_names = {1: "GOOD", 2: "PARAM_FAIL", 3: "FUNC_FAIL"}
    hb_rows = "".join(
        f"<tr><td>H{int(k)} {_esc(hb_names.get(int(k), ''))}</td>"
        f"<td>{int(v)}</td><td>{v / len(results) * 100:.2f}%</td></tr>"
        for k, v in sorted(hb.items()))
    cpk_rows = "".join(
        f"<tr><td>{_esc(r.item)}</td><td>{_esc(r.units)}</td>"
        f"<td>{r.cpk:.2f}</td><td>{r.ppk:.2f}</td><td>{_esc(r.judgement)}</td></tr>"
        for r in cpk_table.sort_values("cpk").head(5).itertuples())
    det_rate = det["detection_rate"] * 100
    diag = float(fa_report["diagonal_hit_rate"]) * 100
    sil = float(fa_report["method"]["silhouette_score"])
    die_t = float(time_report["baseline_s"]) * 1000
    boundary = shmoo_boundary(shmoo_summary)

    cards = [
        ("整体良率（规格限）", f"{y_spec * 100:.2f}%", "guardband 前工艺良率"),
        ("整体良率（测试限）", f"{y_test * 100:.2f}%", f"DPPM {dppm:,.0f}"),
        ("CP 可检出缺陷检出率", f"{det_rate:.2f}%",
         f"{det['detectable_caught']}/{det['detectable_total']}（OFFSET/LEAK/OPEN）"),
        ("guardband 过杀率", f"{det['overkill_rate'] * 100:.2f}%",
         f"{det['overkill']}/{det['clean_total']} 颗无缺陷 die"),
        ("潜伏缺陷逃逸", f"{det['latent_escapes']:,}",
         f"DRIFT/TIMING，需多温/FT/Shmoo"),
        ("STDF 文件", f"{stdf_mb:.1f} MB", f"{n_rec:,} 条记录，V4 二进制"),
        ("最低 Cpk 参数", f"{_esc(cpk_min['item'])}",
         f"Cpk {cpk_min['cpk']:.2f} / Ppk {cpk_min['ppk']:.2f}"),
        ("FA 混淆矩阵对角线", f"{diag:.2f}%", f"轮廓系数 {sil:.4f}（K=5）"),
        ("Shmoo 失效边界", _esc(boundary), "DEF_TIMING 高温重载角"),
        ("单 die 测试时间", f"{die_t:.0f} ms", f"基线 {time_report['baseline_s']:.3f} s"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="card-label">{_esc(label)}</div>'
        f'<div class="card-value">{value}</div>'
        f'<div class="card-sub">{_esc(sub)}</div></div>'
        for label, value, sub in cards)

    sections = []
    for title, path in images.items():
        p = Path(path)
        if p.exists():
            sections.append(
                f'<figure><figcaption>{_esc(title)}</figcaption>'
                f'<img src="data:image/png;base64,{_img_b64(p)}" '
                f'alt="{_esc(title)}"/></figure>')
    imgs_html = "\n".join(sections)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>PMU300 CP 测试报告 — {_esc(lot_id)}</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Hiragino Sans GB",
         "Microsoft YaHei", sans-serif; margin: 0; background: #f5f6f8; color: #222; }}
  header {{ background: #1f3a5f; color: #fff; padding: 28px 36px; }}
  header h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 0; opacity: .8; font-size: 13px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px 48px; }}
  h2 {{ font-size: 17px; margin: 28px 0 12px; border-left: 4px solid #2563eb;
       padding-left: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
          gap: 12px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,.07); }}
  .card-label {{ font-size: 12px; color: #6b7280; }}
  .card-value {{ font-size: 20px; font-weight: 600; margin: 6px 0 4px;
                color: #1f3a5f; word-break: break-all; }}
  .card-sub {{ font-size: 11px; color: #9ca3af; }}
  figure {{ background: #fff; border-radius: 10px; padding: 14px; margin: 0 0 16px;
           box-shadow: 0 1px 3px rgba(0,0,0,.07); }}
  figcaption {{ font-size: 13px; color: #4b5563; margin-bottom: 10px; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 7px 10px; text-align: left; }}
  th {{ background: #f3f4f6; }}
  footer {{ max-width: 1100px; margin: 0 auto; padding: 0 20px 36px;
           font-size: 11px; color: #9ca3af; line-height: 1.7; }}
</style>
</head>
<body>
<header>
  <h1>PMU300 CP 测试报告 · {_esc(lot_id)}</h1>
  <p>程序 {_esc(program)} ｜ 生成日期 {date} ｜ 种子 20261020（全流程可复现）
     ｜ 虚拟 ATE 平台演练（行为模型仿真数据，非真实产线）</p>
</header>
<main>
  <h2>关键 KPI</h2>
  <div class="grid">{cards_html}</div>

  <h2>Hard Bin 分布</h2>
  <table><tr><th>Hard Bin</th><th>计数</th><th>占比</th></tr>{hb_rows}</table>

  <h2>Cpk 最低 Top5（测试限口径，HBin1）</h2>
  <table><tr><th>参数</th><th>单位</th><th>Cpk</th><th>Ppk</th><th>判定</th></tr>{cpk_rows}</table>

  <h2>图表</h2>
  {imgs_html}
</main>
<footer>
  诚实声明：PMU300 为虚构芯片（行为模型）；全部数据来自固定种子蒙特卡洛仿真，
  非真实产线数据；缺陷标签仅用于事后验证（检出率/混淆矩阵），对测试程序不可见；
  仪器层为 pyvisa-sim 仿真（SCPI 命令集与真实仪器一致），未接触真实 ATE 机台。
</footer>
</body>
</html>"""
    out_path.write_text(page, encoding="utf-8")
    return out_path


def run(outputs_dir: Path = DEFAULT_OUT,
        out_path: Path = DEFAULT_OUT / "report.html") -> Path:
    """读取 outputs/ 全部产物，生成报告（独立运行入口）。"""
    kw = collect_default_inputs(outputs_dir)
    return generate_report(out_path, **kw)


if __name__ == "__main__":
    p = run()
    print(f"HTML 报告已生成: {p}（{p.stat().st_size / 1e3:.0f} KB）")
