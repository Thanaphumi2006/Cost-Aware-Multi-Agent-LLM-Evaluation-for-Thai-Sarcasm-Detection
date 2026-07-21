# -*- coding: utf-8 -*-
"""Load keys from a .env file -- so scripts find the key without setting a machine-wide env var

Why it exists: on Windows, setting a User-scope env var does write to the registry
but **already-running processes will not see the new value** (you would have to restart the program)
the .env file is read on every run -> works immediately, no restart needed

`.env` is already in .gitignore -> keys will not leak into git
a value already set in the real environment always **wins** over the .env file (existing values are not overwritten)

Usage: import envload at the top of any script needing the key (a plain import already does the work)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# look for .env in both the Gold/ folder and the repo root
_CANDIDATES = [os.path.join(_HERE, ".env"),
               os.path.join(os.path.dirname(_HERE), ".env")]


def load(paths=None, override=False):
    """read KEY=VALUE line by line · skip blank lines and # · strip surrounding quotes"""
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
