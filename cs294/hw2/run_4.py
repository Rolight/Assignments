from multiprocessing import Process
import subprocess

CartPole_command = ["python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna",
                    "python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna",
                    "python train_pg.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na",
                    "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna",
                    "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna",
                    "python train_pg.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na"]

InvertedPendulum_command = [
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg --exp_name IP_rtg_na",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -dna --exp_name IP_rtg_dna",
]

InvertedPendulum_command_v1 = [
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.001 --exp_name IP_v1_rtg_na_lr_0.001",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.008 --exp_name IP_v1_rtg_na_lr_0.008",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.011 --exp_name IP_v1_rtg_na_lr_0.011",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.015 --exp_name IP_v1_rtg_na_lr_0.015",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 --exp_name IP_v1_rtg_na_lr_0.02",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -l 2 --exp_name IP_v1_rtg_na_layer_2",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -l 3 --exp_name IP_v1_rtg_na_layer_3",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -l 4 --exp_name IP_v1_rtg_na_layer_4",
]

InvertedPendulum_command_v2 = [
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.01 -l 3 --exp_name IP_v2_rtg_na_lr_0.01_layer_3",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.015 -l 3 --exp_name IP_v2_rtg_na_lr_0.015_layer_3",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 3 --exp_name IP_v2_rtg_na_lr_0.02_layer_3",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.025 -l 3 --exp_name IP_v2_rtg_na_lr_0.025_layer_3",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.03 -l 3 --exp_name IP_v2_rtg_na_lr_0.03_layer_3",
]

InvertedPendulum_command_BaseLine = [
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 3 --nn_baseline --exp_name IP_final_rtg_with_baseline",
    "python train_pg.py InvertedPendulum-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 3 --exp_name IP_final_rtg_without_baseline",
]


def run_command(cmd):
    cmd = cmd.split(' ')
    # cmd += ['>', cmd[-1] + '.log']
    subprocess.run(cmd)


all_p = []

for cmd in InvertedPendulum_command_BaseLine:
    p = Process(target=run_command, args=(cmd,))
    all_p.append(p)

for p in all_p:
    p.start()

for p in all_p:
    p.join()
