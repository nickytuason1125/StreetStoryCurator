import json, io, shutil, datetime

p = r'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\cache\catalog.json'
bk = p + '.pre-clear-' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.bak'
shutil.copyfile(p, bk)
d = json.load(io.open(p, encoding='utf-8'))
removed = len(d.get('photos', []))
d['photos'] = []
json.dump(d, io.open(p, 'w', encoding='utf-8'))
print('backed up to', bk)
print('removed', removed, 'old entries - catalog cleared')