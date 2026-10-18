"""Verify every SHA256SUMS entry without importing scientific or training code."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    root = a.root.resolve()
    problems=[]
    entries=[]
    for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ',1)
        target = (root/name).resolve()
        if not target.is_relative_to(root):
            problems.append(name+': outside root')
            continue
        actual=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        entries.append(name)
        if actual != digest:
            problems.append(name+': digest mismatch or missing')
    print(json.dumps({'status':'PASS' if not problems else 'FAIL', 'checked_files':len(entries), 'problems':problems},indent=2))
    return 1 if problems else 0

if __name__ == '__main__':
    raise SystemExit(main())
