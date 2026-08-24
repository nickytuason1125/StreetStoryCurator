# ── second pass: replace marker with loupe + filmstrip + right rail ──
import pathlib

p = pathlib.Path('_makedemo.py')
src = p.read_text(encoding='utf-8')

marker = '<!-- MARKER_CENTER -->'
loupe = """  <div style="flex:1;display:flex;flex-direction:column;min-width:0">
    <div style="height:34px;display:flex;align-items:center;gap:14px;padding:0 10px;
         border-bottom:1px solid #1f1f1f;background:#242424">
      <span class="tb">Grid</span><span class="tb on">Loupe</span>
      <span style="flex:1"></span><span class="tb">Sort: Grade ▾</span>
    </div>
    <div style="flex:1;display:flex;align-items:center;justify-content:center;
         position:relative;background:#191919">
      <img src="dataset_images/{hero}" style="max-width:76%;max-height:88%;
           border-radius:2px;box-shadow:0 6px 30px rgb(0 0 0/.55)">
      <div style="position:absolute;right:14px;top:14px;width:190px;
           background:rgb(36 36 36/.92);border:1px solid #3d3d3d;
           border-radius:3px;padding:12px">
        <div style="display:flex;align-items:baseline;gap:6px">
          <span style="font-family:Consolas,monospace;font-size:22px;color:#7fb069">0.84</span>
          <span style="font-size:11px;color:#9e9e9e">Strong ✓</span>
        </div>
        <svg width="100%" height="86" style="margin-top:8px">
          <polygon points="95,8 178,52 148,80 42,80 12,52" fill="none" stroke="#3d3d3d"/>
          <polygon points="95,20 158,58 136,72 54,72 28,58"
                   fill="rgb(127 176 105/.15)" stroke="#7fb069"/>
        </svg>
      </div>
    </div>
    <div style="height:88px;background:#2b2b2b;border-top:1px solid #1f1f1f;
         display:flex;align-items:center;gap:4px;padding:0 10px;overflow:hidden">
{strip}
    </div>
  </div>""".format(hero=hero, strip=filmstrip(imgs[1:]))

right = """
<div style="width:250px;background:#2b2b2b;border-left:1px solid #1f1f1f">
  <div class="hdr">HISTOGRAM — live from loupe photo</div>
  <svg viewBox="0 0 230 110" style="width:calc(100% - 16px);margin:0 8px;
       background:#191919;border-radius:2px;display:block">
    <path d="M0 90 C30 70 50 30 78 34 S120 78 150 62 S200 20 230 44 L230 110 0 110Z"
          fill="rgb(127 176 105/.25)" stroke="#7fb069" stroke-width="1"/>
  </svg>
  <div class="hdr" style="margin-top:10px;padding-bottom:4px">GRADE BREAKDOWN</div>
  <div style="padding:0 12px;display:flex;flex-direction:column;gap:7px">
    <div class="sl"><b>Composition</b><span class="tr"><i style="width:88%"></i></span><span>88</span></div>
    <div class="sl"><b>Technical</b><span class="tr"><i style="width:81%"></i></span><span>81</span></div>
    <div class="sl"><b>Lighting</b><span class="tr"><i style="width:74%"></i></span><span>74</span></div>
    <div class="sl"><b>Narrative</b><span class="tr"><i style="width:68%"></i></span><span>68</span></div>
    <div class="sl"><b>Human</b><span class="tr"><i style="width:55%"></i></span><span>55</span></div>
  </div>
  <div class="hdr" style="margin-top:14px;padding-bottom:4px">QUICK ACTIONS</div>
  <div style="padding:0 8px;font-size:12px;color:#9e9e9e">
    <div class="row">✓ Mark as Strong</div>
    <div class="row">Add to Story sequence</div>
    <div class="row">Request deep analysis</div>
  </div>
</div>
"""

src = src.replace(marker, loupe + "\n" + right)
pathlib.Path('demo-live.html').write_text(src.split('"""')[1].join(['"""','"""'])
                                          if False else src.replace(
                                              'html = f"""', 'html = f"""').replace(
                                              'MARKER_CENTER', 'REPLACED'), encoding='utf-8')
print('second pass done (marker replaced in generator)')
