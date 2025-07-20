from pathlib import Path
import shutil
import sys


def main(mt5_dir: str = '/opt/mt5') -> None:
    """Copy EA files into the given MetaTrader directory."""
    mt5_path = Path(mt5_dir)
    experts = mt5_path / 'MQL5' / 'Experts'
    experts.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    for name in ['AdaptiveEA.mq5', 'RealtimeEA.mq5']:
        src = repo_root / name
        if src.exists():
            dest = experts / name
            shutil.copy2(src, dest)
            print(f'Copied {src.name} to {dest}')
        else:
            print(f'Source {src} not found')


if __name__ == '__main__':
    mt5_dir = sys.argv[1] if len(sys.argv) > 1 else '/opt/mt5'
    main(mt5_dir)
