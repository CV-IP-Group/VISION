import pathlib
path = pathlib.Path.home() / '.cache' / 'torch' / 'hub' / 'intel-isl_MiDaS_master' / 'hubconf.py'
print('hubconf exists', path.exists())
if path.exists():
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('def ') and stripped.endswith('):'):
            name = stripped.split('def ')[1].split('(')[0]
            print(name)
