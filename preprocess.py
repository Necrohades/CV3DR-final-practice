from pathlib import Path

for path in Path("archive/cup/labels").iterdir():
    output = []

    with path.open("r") as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = "0"
            output.append(" ".join(parts))
    with path.open("w") as file:
        file.write("\n".join(output))
