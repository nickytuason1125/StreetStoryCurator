"""One-shot RAM hog report — top processes over 150 MB."""
import psutil

procs = []
for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
    try:
        mi = p.info["memory_info"]
        if mi and mi.rss > 150 * (1 << 20):
            cl = " ".join(p.info["cmdline"] or [])[:70]
            procs.append((mi.rss / (1 << 20), p.info["name"], p.info["pid"], cl))
    except Exception:
        pass
procs.sort(reverse=True)
with open("hog_report.txt", "w", encoding="utf-8") as f:
    f.write(f"{len(procs)} processes over 150 MB:\n")
    for rss, name, pid, cl in procs:
        f.write(f"{rss:7.0f} MB  {name} (pid {pid})  {cl}\n")
    vm = psutil.virtual_memory()
    f.write(f"\nphysical free: {vm.available/(1<<30):.2f} GB\n")
print("written: hog_report.txt")
