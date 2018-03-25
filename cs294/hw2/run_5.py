from multiprocessing import Process
import subprocess

HalfCheetah_command = [
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC",
]

HalfCheetah_command_v1 = [
    # Baseline
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_base",
    # configure lr
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.01 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_lr_0.01",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.025 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_lr_0.025",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.03 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_lr_0.03",

    # configure layer
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 2 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_layer_2",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 4 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_layer_4",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 1000 -e 5 -rtg -lr 0.02 -l 5 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_layer_5",

    # large batch
    "python train_pg.py HalfCheetah-v2 -n 100 -b 2000 -e 5 -rtg -lr 0.02 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_lb_2000",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 5000 -e 5 -rtg -lr 0.02 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v1_lb_5000",

]

# lr=0.025, 5 layer, 5000 batch size work best
HalfCheetah_command_v2 = [
    # try larger batch size
    "python train_pg.py HalfCheetah-v2 -n 100 -b 5000 -e 5 -rtg -lr 0.025 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v2_lb_5000",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 10000 -e 5 -rtg -lr 0.025 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v2_lb_10000",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 15000 -e 5 -rtg -lr 0.025 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v2_lb_15000",
    "python train_pg.py HalfCheetah-v2 -n 100 -b 20000 -e 5 -rtg -lr 0.025 -l 3 -ep 150 --discount 0.9 --nn_baseline --exp_name HC_v2_lb_20000",
]


def run_command(cmd):
    cmd = cmd.split(' ')
    # cmd += ['>', cmd[-1] + '.log']
    subprocess.run(cmd)


all_p = []

for cmd in HalfCheetah_command_v2:
    p = Process(target=run_command, args=(cmd,))
    all_p.append(p)

for p in all_p:
    p.start()

for p in all_p:
    p.join()
