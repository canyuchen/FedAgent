import pathlib, sys
try:
    import yaml
except ImportError:
    print("PyYAML not in this interpreter; trying ruamel/omegaconf fallback")
    sys.exit(2)
bad = []
files = list(pathlib.Path("fedagent/config").rglob("*.yaml"))
for f in files:
    try:
        yaml.safe_load(f.read_text())
    except Exception as e:
        bad.append(f"{f}: {e}")
print(f"parsed {len(files)} yaml files under fedagent/config/")
print(f"  paper/: {len(list(pathlib.Path('fedagent/config/paper').rglob('*.yaml')))}")
if bad:
    print(f"INVALID ({len(bad)}):"); [print(' ', b) for b in bad]; sys.exit(1)
print("ALL valid YAML ✅")
