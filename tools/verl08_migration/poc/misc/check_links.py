import re, pathlib
root = pathlib.Path("fedagent")
md = list(root.glob("*/README.md")) + list(root.glob("docs/*.md")) + [root/"README.md"]
link = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
broken = []
for f in md:
    for m in link.finditer(f.read_text()):
        t = m.group(1).strip()
        if t.startswith(("http://","https://","#","mailto:")): continue
        t = t.split("#")[0]
        if not t: continue
        if not (f.parent / t).resolve().exists():
            broken.append(f"{f}: -> {t}")
print(f"checked {len(md)} files")
if broken:
    print(f"BROKEN ({len(broken)}):"); [print(" ", b) for b in broken]
else:
    print("ALL relative links resolve ✅")
