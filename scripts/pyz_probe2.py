import os, glob
base = 'dist/FirstCut'
print('root entries:')
for e in sorted(os.listdir(base)):
    p = os.path.join(base, e)
    print('  ', e, '(dir)' if os.path.isdir(p) else str(os.path.getsize(p)))
intern = os.path.join(base, '_internal')
pyzish = [e for e in os.listdir(intern) if e.endswith(('.pyz', '.pkg', '.zip'))]
print('pyz-like in _internal:', pyzish)
r = os.path.join(intern, 'routers')
print('routers dir exists:', os.path.isdir(r))
if os.path.isdir(r):
    print('routers files:', len(os.listdir(r)))
