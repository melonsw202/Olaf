"""restart 전에 모델 checkpoint 안의 learning rate를 바꾸는 스크립트.
restart: true이면 checkpoint의 optimizer 상태가 input.json의 learning_rate를 덮어쓰므로 필요함.
사용: $L/env/bin/python set_ckpt_lr.py <원본 model_ft.pkl> <새 model_ft.pkl> <lr>
원본 파일은 읽기만 하고, 결과는 새 경로에 저장 (같은 경로 지정 금지)."""
import os, sys, torch
src, dst, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
if os.path.abspath(src) == os.path.abspath(dst):
    sys.exit("원본과 다른 경로에 저장하세요")
c = torch.load(src, map_location="cpu", weights_only=False)
for g in c["opt_state"]["param_groups"]:
    g["lr"] = lr
s = c.get("scheduler")
if isinstance(s, dict) and "_last_lr" in s:
    s["_last_lr"] = [lr] * len(s["_last_lr"])
torch.save(c, dst)
print("ckpt lr ->", [g["lr"] for g in torch.load(dst, map_location="cpu", weights_only=False)["opt_state"]["param_groups"]])
