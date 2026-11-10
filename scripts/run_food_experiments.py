"""Run three predeclared controls, select on validation, then evaluate test."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foodcal.data import file_hash, load_config
from foodcal.training import train, evaluate


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--output', default='runs/food10_seed42')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--wait-for-data', action='store_true', help='Wait for the already-running dataset downloader to release its lock')
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(args.config)
    manifest = Path(config['data']['manifest'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This experiment suite is already running')
        if not manifest.is_file() and args.wait_for_data:
            save(out / 'status.json', {'state': 'waiting_for_data', 'manifest': str(manifest)})
            print('Waiting for the existing Food-101 download and preparation...', flush=True)
            with Path('data/raw/.food101-download.lock').open('a') as data_lock:
                fcntl.flock(data_lock, fcntl.LOCK_EX)
        if not manifest.is_file():
            save(out / 'status.json', {'state': 'failed', 'error': 'Dataset preparation did not produce a manifest'})
            raise SystemExit('Prepare the dataset first: scripts/download_food101.py --prepare')
        conditions = [('A_frozen', 'frozen', False), ('B_last_block', 'last_block', False),
                      ('C_last_block_color', 'last_block', True)]
        protocol = {'config': config, 'manifest_sha256': file_hash(manifest),
                    'pretrained_sha256': file_hash(config['model']['pretrained_weights']),
                    'implementation_sha256': {name: file_hash(ROOT / 'foodcal' / name)
                                              for name in ('model.py', 'training.py', 'metrics.py')},
                    'suite_implementation_sha256': file_hash(Path(__file__)),
                    'conditions': conditions, 'selection': 'Highest validation macro-F1; earliest condition breaks ties',
                    'evaluation': 'Select deployment before opening test; evaluate all fixed controls once',
                    'scope': 'One training seed; not a repeated-seed estimate', 'device': args.device}
        protocol = json.loads(json.dumps(protocol))
        protocol_path = out / 'suite_protocol.json'
        if protocol_path.exists():
            if json.loads(protocol_path.read_text()) != protocol:
                raise ValueError('Protocol changed; use a new output directory')
        else:
            save(protocol_path, protocol)
        summaries = {}
        try:
            preflight = out / 'real_image_preflight.json'
            if not preflight.exists():
                from scripts.check_food_model import check
                check(config, preflight, args.device)
            for name, mode, augment in conditions:
                destination = out / name
                save(out / 'status.json', {'state': 'training', 'condition': name,
                     'completed': list(summaries), 'updated_utc': datetime.now(timezone.utc).isoformat()})
                summary_path = destination / 'summary.json'
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    run_protocol = json.loads((destination / 'protocol.json').read_text())
                    if (summary['manifest_sha256'] != protocol['manifest_sha256'] or
                            summary['mode'] != mode or summary['augment'] != augment or
                            run_protocol['config'] != config or
                            run_protocol['pretrained_sha256'] != protocol['pretrained_sha256'] or
                            run_protocol['implementation_sha256'] != protocol['implementation_sha256']):
                        raise ValueError('Completed run does not match the suite protocol')
                else:
                    if destination.exists():
                        raise ValueError(f'Incomplete run preserved at {destination}; use a new suite output to retry')
                    summary = train(manifest, config, destination, mode, augment, args.device)
                summaries[name] = summary
            selected = max(summaries, key=lambda name: summaries[name]['macro_f1'])
            selection = {'condition': selected, 'basis': 'validation_macro_f1',
                         'score': summaries[selected]['macro_f1'], 'test_used_for_selection': False,
                         'checkpoint_sha256': file_hash(out / selected / 'checkpoint.pt'),
                         'validation_scores': {name: value['macro_f1'] for name, value in summaries.items()}}
            # Persist the choice before any held-out test image is evaluated.
            save(out / 'selection.json', selection)
            results = []
            for name, mode, augment in conditions:
                save(out / 'status.json', {'state': 'testing', 'condition': name, 'selected': selected})
                checkpoint = out / name / 'checkpoint.pt'
                test_output = out / (name + '_test')
                metrics_path = test_output / 'metrics.json'
                completion = test_output / 'completion.json'
                if completion.exists():
                    checksums = json.loads(completion.read_text())
                    if any(file_hash(test_output / name) != digest for name, digest in checksums.items()):
                        raise ValueError('Completed evaluation artifacts have changed')
                    metrics = json.loads(metrics_path.read_text())
                    if (metrics['checkpoint_sha256'] != file_hash(checkpoint) or
                            metrics['manifest_sha256'] != protocol['manifest_sha256']):
                        raise ValueError('Existing evaluation does not match the checkpoint')
                else:
                    if test_output.exists():
                        raise ValueError(f'Incomplete evaluation preserved at {test_output}; inspect it before retrying')
                    metrics = evaluate(manifest, checkpoint, test_output, 'test', args.device)
                    save(completion, {name: file_hash(test_output / name) for name in
                                      ('metrics.json', 'predictions.csv', 'confusion_matrix.csv')})
                summary = summaries[name]
                results.append({'condition': name, 'mode': mode, 'augment': augment,
                                'best_epoch': summary['best_epoch'], 'epochs': summary['epochs_completed'],
                                'validation_accuracy': summary['accuracy'], 'validation_macro_f1': summary['macro_f1'],
                                'test_accuracy': metrics['accuracy'], 'test_macro_f1': metrics['macro_f1'],
                                'training_seconds': summary['elapsed_seconds']})
            target = Path('models/food10/best.pt')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_hash(target) != selection['checkpoint_sha256']:
                raise ValueError('A different deployment checkpoint exists; it was preserved')
            if not target.exists():
                temporary = target.with_suffix('.tmp')
                shutil.copyfile(out / selected / 'checkpoint.pt', temporary)
                temporary.replace(target)
            save(target.with_suffix('.json'), {**selection, 'suite': str(out), 'manifest_sha256': protocol['manifest_sha256']})
            save(out / 'comparison.json', {'selection': selection, 'results': results, 'scope': protocol['scope']})
            save(out / 'status.json', {'state': 'completed', 'selected': selected, 'checkpoint': str(target)})
            print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        except BaseException as exc:
            save(out / 'status.json', {'state': 'failed' if isinstance(exc, Exception) else 'interrupted',
                 'error_type': type(exc).__name__, 'error': str(exc), 'completed': list(summaries)})
            raise


if __name__ == '__main__':
    main()
