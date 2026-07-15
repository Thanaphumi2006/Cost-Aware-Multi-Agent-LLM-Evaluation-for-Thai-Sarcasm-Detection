# -*- coding: utf-8 -*-
"""การเทียบที่ยุติธรรม: ทุกระบบ vs baseline ที่ "tune แล้ว" (ไม่ใช่ baseline ดิบ)

ทำไมต้องมีไฟล์นี้ (ประเด็นสำคัญที่สุดของโปรเจกต์):
  compare_systems.py เทียบทุกระบบกับ baseline @argmax (F1 0.690) ซึ่ง "ทิ้ง logprob" ที่จ่ายเงินซื้อมาแล้ว
  -> เท่ากับให้ multi-agent สู้กับคู่ต่อสู้ที่ถูกมัดมือ -> ให้เครดิต multi-agent เกินจริง
  คู่เทียบที่ถูกต้องคือ baseline + threshold (F1 0.725, ราคาเท่าเดิม $0.094, ไม่เพิ่ม call)

ไฟล์นี้ไม่ยิง API เลย -- ใช้ pred ที่บันทึกไว้แล้วทุกระบบ
รัน: python compare_fair.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = "multiagent_preds_gpt_threshold.csv"   # baseline ที่ tune แล้ว = คู่เทียบที่ยุติธรรม
REF_NAME = "baseline+threshold"
N_BOOT = 5000
RNG = np.random.default_rng(0)

# ราคาต่อระบบ (จาก RESULTS.md) ไว้เตือนว่า "จ่ายเพิ่มเพื่ออะไร"
COST = {"baseline": 0.094, "baseline+threshold": 0.094, "conservative": 0.169,
        "v1aggressive": 0.157, "debate": 0.695, "hybrid": 0.407, "cascade": 0.124}


def load(path):
    d = pd.read_csv(path, dtype=str).fillna("")
    d["label"] = d["label"].str.strip()
    d = d[d["pred"].isin(["0", "1"])]
    return d.set_index("text")[["label", "pred"]]


def f1(y, p):
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    ref = load(os.path.join(HERE, REF_FILE))
    systems = {"baseline": "baseline_preds_gpt.csv"}
    for p in sorted(glob.glob(os.path.join(HERE, "multiagent_preds_gpt_*.csv"))):
        nm = os.path.basename(p).replace("multiagent_preds_gpt_", "").replace(".csv", "")
        if nm == "threshold":
            continue                      # อันนี้คือ ref เอง
        systems[nm] = p
    systems["wangchanberta"] = "wangchanberta_preds.csv"

    print(f"คู่เทียบ (ยุติธรรม) = {REF_NAME}: F1 {f1((ref['label']=='1').values,(ref['pred']=='1').values):.3f} "
          f"| ${COST[REF_NAME]:.3f} | 127 calls\n")
    print(f"{'ระบบ':<16}{'F1':>6}{'ต่าง':>8}{'95% CI':>20}{'P(ไม่ดีกว่า)':>13}{'McNemar':>12}{'ราคา':>8}")
    print("-" * 84)

    for name, fn_ in systems.items():
        s = load(os.path.join(HERE, fn_))
        common = ref.index.intersection(s.index)
        r, ss = ref.loc[common], s.loc[common]
        y = (r["label"].values == "1")
        pr = (r["pred"].values == "1")      # ref (tuned baseline)
        ps = (ss["pred"].values == "1")     # ระบบที่กำลังเทียบ
        n = len(common)

        diffs = np.array([f1(y[i], ps[i]) - f1(y[i], pr[i])
                          for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))])
        d0 = f1(y, ps) - f1(y, pr)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_not = (diffs <= 0).mean() * 100
        win = int(((ps == y) & (pr != y)).sum())   # ระบบถูก-ref ผิด
        los = int(((pr == y) & (ps != y)).sum())   # ref ถูก-ระบบผิด
        c = COST.get(name, float("nan"))
        star = "  <-" if (lo <= 0 <= hi) else ""    # CI คร่อม 0 = แยกจาก ref ไม่ออก
        print(f"{name:<16}{f1(y,ps):>6.3f}{d0:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{p_not:>10.0f}%"
              f"{f'{win}-{los}':>12}{f'${c:.3f}':>8}{star}")

    print("\nอ่านยังไง:")
    print("  'ต่าง' = F1(ระบบ) - F1(baseline+threshold) ; บวก = ระบบดีกว่าคู่เทียบที่ยุติธรรม")
    print("  '<-' = CI คร่อม 0 = แยกออกจาก baseline ที่ tune แล้วไม่ได้ (จ่ายเพิ่มเพื่อ noise)")
    print("  McNemar = (ระบบถูก-baseline+threshold ผิด) - (กลับกัน) ; ใกล้เสมอ = ไม่มีข้อได้เปรียบจริง")


if __name__ == "__main__":
    main()
