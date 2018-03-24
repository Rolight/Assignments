from multiprocessing import Process
import subprocess

command = ["python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna",
           "python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna",
           "python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na",
           "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna",
           "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna",
           "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na"]


def run_command(cmd):
    cmd = cmd.split(' ')
    # cmd += ['>', cmd[-1] + '.log']
    subprocess.run(cmd)


all_p = []

for cmd in command:
    p = Process(target=run_command, args=(cmd,))
    all_p.append(p)

for p in all_p:
    p.start()

for p in all_p:
    p.join()
