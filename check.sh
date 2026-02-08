#!/usr/bin/env bash
set -euo pipefail

CPU_XML="/home/kaiii/Gem5McPatParser/mcpat-cpu.xml"
CIM_XML="/home/kaiii/Gem5McPatParser/mcpat-cim.xml"

if [[ ! -f "$CPU_XML" || ! -f "$CIM_XML" ]]; then
  echo "missing input:"
  [[ ! -f "$CPU_XML" ]] && echo "  not found: $CPU_XML"
  [[ ! -f "$CIM_XML" ]] && echo "  not found: $CIM_XML"
  exit 1
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

norm_one () {
  local in="$1"
  local out="$2"

  # 1) extract param/stat with component path
  python3 - "$in" > "$out.raw" <<'PY'
import sys
import xml.etree.ElementTree as ET

xml_path = sys.argv[1]
tree = ET.parse(xml_path)
root = tree.getroot()

if root.tag == "document":
    comp = root.find("./component")
    if comp is None:
        raise SystemExit("no <component> under <document>")
    root = comp

def comp_label(elem):
    n = elem.get("name", "")
    i = elem.get("id", "")
    if n and i: return f"{n}:{i}"
    if n: return f"{n}:"
    if i: return f":{i}"
    return ":"

def walk(comp, path):
    cur = path + [comp_label(comp)]
    cur_path = "/".join(cur)

    for p in comp.findall("./param"):
        name = p.get("name", "")
        val  = p.get("value", "")
        print(f"param|{cur_path}|{name}={val}")
    for s in comp.findall("./stat"):
        name = s.get("name", "")
        val  = s.get("value", "")
        print(f"stat|{cur_path}|{name}={val}")

    for c in comp.findall("./component"):
        walk(c, cur)

walk(root, [])
PY

  # 2) normalize numbers + strip CR
  python3 - "$out.raw" > "$out.tmp" <<'PY'
import sys, re, math

inp = sys.argv[1]
num_re = re.compile(r'([-+]?\d+\.\d+|[-+]?\d+)(?![\w\.])')

def norm_num(m):
    s = m.group(1)
    try:
        x = float(s)
        if not math.isfinite(x):
            return "0"
        # 6dp then trim
        t = f"{x:.6f}"
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        return t
    except:
        return s

with open(inp, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        # normalize only the value part after '='
        if "=" in line:
            a, b = line.split("=", 1)
            b = num_re.sub(norm_num, b)
            print(a + "=" + b)
        else:
            print(line)
PY

  # 3) FORCE C-locale sort (comm requires sorted)
  LC_ALL=C sort -u "$out.tmp" > "$out"
}

CPU_N="$TMPDIR/cpu.norm"
CIM_N="$TMPDIR/cim.norm"

norm_one "$CPU_XML" "$CPU_N"
norm_one "$CIM_XML" "$CIM_N"

# verify sorted (if not, show first offending region)
if ! LC_ALL=C sort -c "$CPU_N" 2>/dev/null; then
  echo "CPU_N not sorted; showing sample:"
  head -n 5 "$CPU_N"
  exit 2
fi
if ! LC_ALL=C sort -c "$CIM_N" 2>/dev/null; then
  echo "CIM_N not sorted; showing sample:"
  head -n 5 "$CIM_N"
  exit 2
fi

echo "===== 1) Unified diff (with component path) ====="
diff -u "$CPU_N" "$CIM_N" || true
echo

echo "===== 2) Only in CPU (top 200) ====="
LC_ALL=C comm -23 "$CPU_N" "$CIM_N" | head -n 200 || true
echo

echo "===== 3) Only in CIM (top 200) ====="
LC_ALL=C comm -13 "$CPU_N" "$CIM_N" | head -n 200 || true
echo

echo "===== 4) Same key different value (top 200) ====="
python3 - "$CPU_N" "$CIM_N" <<'PY'
import sys

def load(p):
    d={}
    with open(p,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            # param|path|name=value
            try:
                t, path, nv = line.split('|',2)
                name, val = nv.split('=',1)
                d[(t,path,name)] = val
            except:
                pass
    return d

cpu = load(sys.argv[1])
cim = load(sys.argv[2])

diffs=[]
for k in cpu.keys() & cim.keys():
    if cpu[k] != cim[k]:
        diffs.append((k, cpu[k], cim[k]))

diffs.sort(key=lambda x: (x[0][0], x[0][1], x[0][2]))

for (t,path,name), v1, v2 in diffs[:200]:
    print(f"{t}|{path}|{name} | CPU={v1} | CIM={v2}")
print(f"\nTotal diffs: {len(diffs)}")
PY

echo
echo "Done."