"""Fetch and verify the official torchvision MobileNetV3-Small initialization."""
import hashlib
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'
SHA256 = '047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f'


def main():
    os.chdir(ROOT)
    target = Path('models/pretrained/mobilenet_v3_small-047dcff4.pth')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Existing weights do not match the official file; preserved for inspection')
        print('Official ImageNet initialization already present and verified')
        return
    partial = target.with_suffix('.part')
    with urllib.request.urlopen(URL, timeout=30) as source, partial.open('xb') as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Downloaded weights failed SHA-256 verification')
    partial.replace(target)
    print('Official ImageNet initialization downloaded and verified; not a trained food classifier')


if __name__ == '__main__':
    main()
