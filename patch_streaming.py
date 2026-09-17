from pathlib import Path
import re

p = Path("/opt/chatterbox-flash/chatterbox_flash/model.py")
s = p.read_text(encoding="utf-8")

# Add optional block_callback to the T3 generate() signature.
pattern = r'(backend:\s*Literal\["auto",\s*"flashinfer",\s*"torch"\]\s*=\s*"auto",\s*\n)(\s*\)\s*->\s*Tensor:)'
m = re.search(pattern, s)
if not m:
    raise RuntimeError(
        "Could not find the upstream T3 generate() backend signature. "
        "Upstream source changed; review patch_streaming.py."
    )

indent = re.match(r"(\s*)", m.group(1)).group(1)
replacement = m.group(1) + indent + "block_callback=None,\n" + m.group(2)
s = s[:m.start()] + replacement + s[m.end():]

# Find the committed-block point used by the current upstream implementation.
needle = "shift_ctx = fin_hidden[:, bl - 1 : bl, :].clone()"
idx = s.find(needle)
if idx < 0:
    raise RuntimeError(
        "Could not find the upstream block commit point. "
        "Upstream source changed; review patch_streaming.py."
    )

line_end = s.find("\n", idx)
line_start = s.rfind("\n", 0, idx) + 1
line = s[line_start:line_end]
base_indent = line[:len(line) - len(line.lstrip())]

hook = (
    "\n"
    + base_indent + "# WebSocket streaming hook: current block is committed.\n"
    + base_indent + "if block_callback is not None:\n"
    + base_indent + "    block_callback(\n"
    + base_indent + "        xt[:B_usr, :be_].clone(),\n"
    + base_indent + "        b_idx,\n"
    + base_indent + "        b_idx == num_blocks - 1,\n"
    + base_indent + "    )\n"
)

s = s[:line_end] + hook + s[line_end:]
p.write_text(s, encoding="utf-8")

print("Applied block_callback streaming patch to:", p)
