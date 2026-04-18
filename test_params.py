import paramiko
import sys

HOST = "172.29.253.162"
USER = "ab"
PASS = "ab@123"
PYTHON = "/home/ab/anaconda3/envs/yww_yolo/bin/python"
REMOTE_DIR = "/media/yww/lwd_detr_project"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

for w in [0.40, 0.35, 0.30]:
    yaml_cmd = f"sed -i 's/  lwd: \\[1.00, 0.45, 1024\\]/  lwd: [1.00, {w}, 1024]/' {REMOTE_DIR}/configs/lwd-detr.yaml"
    client.exec_command(yaml_cmd)
    code = f"""
import sys
sys.path.insert(0, '{REMOTE_DIR}')
import lwd_detr.patch
from ultralytics import RTDETR
model = RTDETR('{REMOTE_DIR}/configs/lwd-detr.yaml')
print(f'width={w}, params={{sum(p.numel() for p in model.model.parameters())/1e6:.2f}}M')
"""
    remote_path = f"{REMOTE_DIR}/test_param.py"
    sftp = client.open_sftp()
    with sftp.file(remote_path, 'w') as f:
        f.write(code)
    sftp.close()
    stdin, stdout, stderr = client.exec_command(f"cd {REMOTE_DIR} && {PYTHON} test_param.py")
    out = stdout.read().decode('utf-8', errors='replace').strip()
    sys.stdout.buffer.write((out + '\n').encode('utf-8', errors='replace'))

client.close()
