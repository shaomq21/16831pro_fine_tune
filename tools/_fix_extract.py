from pathlib import Path

p = Path(__file__).with_name("extract_pi05_vision_features.py")
lines = p.read_text().splitlines()
lines[22] = " _REPO_ROOT = _SCRIPT_DIR" + chr(46) + "parent"
lines[51] = "        .replace(" + chr(34) + " " + chr(34) + ", " + chr(34) + "_" + chr(34) + ")"
p.write_text("\n".join(lines) + "\n")
print(lines[22])
print(lines[51])
