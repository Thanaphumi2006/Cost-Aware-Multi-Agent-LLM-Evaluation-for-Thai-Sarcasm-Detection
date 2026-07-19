# -*- coding: utf-8 -*-
"""โหลดคีย์จากไฟล์ .env -- ให้สคริปต์หาคีย์เจอโดยไม่ต้องตั้ง env var ทั้งเครื่อง

ทำไมต้องมี: บน Windows การตั้ง env var แบบ User scope เขียนลง registry ก็จริง
แต่ **โปรเซสที่เปิดค้างอยู่แล้วจะไม่เห็นค่าใหม่** (ต้องปิดเปิดโปรแกรมใหม่)
ไฟล์ .env อ่านตอนรันทุกครั้ง -> ใช้ได้ทันที ไม่ต้องรีสตาร์ตอะไร

`.env` อยู่ใน .gitignore อยู่แล้ว -> คีย์ไม่หลุดขึ้น git
ค่าที่ตั้งไว้ใน environment จริงจะ **ชนะ** ไฟล์ .env เสมอ (ไม่เขียนทับของที่มีอยู่)

ใช้: import envload ไว้บนสุดของสคริปต์ที่ต้องใช้คีย์ (import เฉยๆ ก็ทำงานแล้ว)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# มองหา .env ทั้งในโฟลเดอร์ Gold/ และ root ของ repo
_CANDIDATES = [os.path.join(_HERE, ".env"),
               os.path.join(os.path.dirname(_HERE), ".env")]


def load(paths=None, override=False):
    """อ่าน KEY=VALUE ทีละบรรทัด · ข้ามบรรทัดว่างและ # · ลอก quote ออกให้"""
    found = []
    for p in (paths or _CANDIDATES):
        if not os.path.exists(p):
            continue
        found.append(p)
        with open(p, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and (override or not os.environ.get(k)):
                    os.environ[k] = v
    return found


load()
