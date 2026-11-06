"""main_gpu.py를 그대로 실행하면서 GPU 최대 메모리를 30초마다 출력.
사용: python mem_wrap.py <main_gpu.py> <input.json>"""
import sys, time, threading, signal, runpy, torch

def report(tag):
    if torch.cuda.is_available():
        a = torch.cuda.max_memory_allocated() / 2**30
        r = torch.cuda.max_memory_reserved() / 2**30
        print(f"[mem] {tag} {time.strftime('%H:%M:%S')} max_alloc={a:.2f}GiB max_reserved={r:.2f}GiB", flush=True)

def loop():
    while True:
        time.sleep(30)
        report("periodic")

def on_term(sig, frame):
    report("sigterm")
    sys.exit(1)

signal.signal(signal.SIGTERM, on_term)
threading.Thread(target=loop, daemon=True).start()
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    report("final")
