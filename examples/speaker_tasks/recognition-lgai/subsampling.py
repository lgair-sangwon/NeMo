import argparse
import json

import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", default="/data/suhsw1210/labs/nemo-lab/data/train.json")
    parser.add_argument("--n", type=int, default=300)  # avg of {vox2: 186, cn: 368, tube: 880}
    args = parser.parse_args()

    # load data
    print("[1] wait for load list...")
    data_list = [json.loads(x) for x in open(args.data_path).read().splitlines()]
    df = pd.DataFrame(data_list)
    df["dataset"] = df.label.str.split("-").str[0]
    df_group = df.groupby("dataset")
    for dataset, group in df_group:
        print(dataset)
        print(len(group) / len(set(group.label)))

    df_tts = df_group.get_group("tts_multi_speaker")
    tts_label_group = df_tts.groupby("label")

    # random sample
    tts_label_group = df.groupby("label")
    sampled_df_list = []
    for label, _group in tqdm(
        tts_label_group, total=len(tts_label_group), desc=f"[3] subsampling by {args.n} for each speakers"
    ):
        if len(_group) > args.n:
            sampled_df_list.append(_group.sample(n=args.n))
        else:
            sampled_df_list.append(_group)
    sampled_df = pd.concat(sampled_df_list)
    print(f"Sample counts: {len(df_tts)} -> {len(sampled_df)}")
    sampled_df = sampled_df[['audio_filepath', 'offset', 'duration', 'label']].sort_values(['audio_filepath'])

    df_vc_libri = df[df.dataset != "tts_multi_speaker"]
    df_vc_libri = df_vc_libri[["audio_filepath", "offset", "duration", "label"]].sort_values("audio_filepath")

    output_df = pd.concat([df_vc_libri, sampled_df])
    with open("dev-even.json", "w") as f:
        for _, l in tqdm(output_df.iterrows(), total=len(output_df), desc="writing.."):
            json.dump(l.to_dict(), f)
            f.write('\n')
