import glob, zipfile, os
a = glob.glob('dist/FirstCut/_internal/*.pyz')
print('pyz:', a)
z = zipfile.ZipFile(a[0])
names = [n for n in z.namelist() if 'extras' in n.lower() or 'routers' in n]
print(len(names), 'router entries in PYZ')
for n in names[:15]:
    print('  ', n)
loose = 'dist/FirstCut/_internal/routers/extras.py'
print('loose extras exists:', os.path.exists(loose))
if os.path.exists(loose):
    src = open(loose, encoding='utf-8', errors='replace').read()
    print('loose has models/pull decorator:', 'api/models/pull' in src)
