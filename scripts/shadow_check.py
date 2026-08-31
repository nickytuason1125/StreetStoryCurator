import os, time
for cand in ['src/routers', 'routers']:
    p = os.path.join('.', cand)
    print(cand, 'exists:', os.path.isdir(p))
    if os.path.isdir(p):
        ex = os.path.join(p, 'extras.py')
        if os.path.exists(ex):
            src = open(ex, encoding='utf-8', errors='replace').read()
            print(' ', ex, 'has models/pull:', 'api/models/pull' in src, ' mtime:', time.ctime(os.path.getmtime(ex)))
