import argparse
from glob import glob

import torch


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", help="체크포인트가 저장된 폴더 경로")
    parser.add_argument("--output_name")
    parser.add_argument("--epochs", nargs="+", type=int, default=[21, 22, 23, 24], help="사용할 체크포인트의 에포크 번호 리스트")
    args = parser.parse_args()

    flist = glob(f"{args.ckpt_root}/*.ckpt")

    # get valid checkpoing list
    ckpt_list = []
    for path in flist:
        epoch = path.split("=")[-1].replace(".ckpt", "")
        if "-last" in epoch:
            continue
        if int(epoch) in args.epochs:
            ckpt_list.append(path)

    # make averaged weights
    checkpoint_avg = None
    n_models = 0
    for checkpoint_path in sorted(ckpt_list)[1:-2]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint_avg is None:
            checkpoint_avg = checkpoint['state_dict']
            n_models += 1
            continue

        for k, v in checkpoint['state_dict'].items():
            checkpoint_avg[k] += v.type_as(checkpoint_avg[k])
        n_models += 1

    print(f'Take average for {n_models} models')
    for k, v in checkpoint_avg.items():
        checkpoint_avg[k] = torch.div(v, n_models)

    # utilze the last checkpoint as a template
    checkpoint["state_dict"] = checkpoint_avg

    torch.save(checkpoint, args.output_name)
