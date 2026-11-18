"""Record source hashes in configs/SOURCE_MANIFEST.json before a new experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = {}
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.name == 'SOURCE_MANIFEST.json':
            continue
        if any(part in ('.git', '__pycache__', 'runs', 'work', 'data', '.venv', 'venv') for part in relative.parts):
            continue
        if path.suffix in ('.pt', '.pth', '.pyc', '.npy', '.npz', '.zip'):
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024), b''):
                digest.update(block)
        files[relative.as_posix()] = digest.hexdigest()
    target = ROOT/'configs'/'SOURCE_MANIFEST.json'
    target.write_text(json.dumps(dict(format_version=1, files=files), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(manifest=target.relative_to(ROOT).as_posix(), source_files=len(files))))


if __name__ == '__main__':
    main()
