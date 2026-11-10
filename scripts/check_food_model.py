"""Resource preflight with a real training image and official ImageNet model."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(config, output, device='auto'):
    from PIL import Image
    import torch
    from torchvision.models import mobilenet_v3_small
    from foodcal.data import file_hash
    from foodcal.model import image_transform, resolve_device

    manifest = Path(config['data']['manifest'])
    with manifest.open() as handle:
        row = next(row for row in csv.DictReader(handle) if row['split'] == 'train')
    path = manifest.parent / row['image_path']
    if file_hash(path) != row['sha256']:
        raise ValueError('The real preflight image does not match the manifest')
    target = resolve_device(device)
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(config['model']['pretrained_weights'], map_location='cpu', weights_only=True))
    model.to(target).eval()
    with Image.open(path) as image:
        image_size = list(image.size)
        tensor = image_transform(config['model']['input_size'])(image).unsqueeze(0).to(target)
    def sync():
        if target.type == 'mps': torch.mps.synchronize()
        elif target.type == 'cuda': torch.cuda.synchronize()
    sync()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
    sync()
    elapsed = time.perf_counter() - started
    if list(logits.shape) != [1, 1000] or not bool(logits.isfinite().all().cpu()):
        raise ValueError('Official pretrained model did not produce finite ImageNet logits')
    result = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'sample_id': row['sample_id'], 'source_split': 'train', 'original_image_size': image_size,
              'image_sha256': row['sha256'], 'device': str(target), 'input_shape': list(tensor.shape),
              'output_shape': list(logits.shape), 'cold_forward_seconds': elapsed,
              'pretrained_sha256': file_hash(config['model']['pretrained_weights']),
              'scope': 'Real training-image resource preflight with official ImageNet1000 model; not food accuracy or a trained food head',
              'timing_scope': 'Single synchronized cold model forward; excludes decode and preprocessing'}
    if target.type == 'mps':
        result['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
        result['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        result['memory_scope'] = 'End-of-forward allocation snapshot; not peak memory'
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    from foodcal.data import load_config
    os.chdir(ROOT)
    print(json.dumps(check(load_config(), 'runs/setup/real_image_preflight.json'), ensure_ascii=False, indent=2))
