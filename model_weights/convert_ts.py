import torch
import argparse


def convert_ts(input_file_name, output_file_name):
    ckpt = torch.load(input_file_name, map_location="cpu")
    state_dict = ckpt["model"]
    prefixes = ["modelStudent.", "modelTeacher."]

    new_state_dict = {}
    for prefix in prefixes:
        for k, v in state_dict.items():
            if "queue" in k: continue
            k = prefix + k
            new_state_dict[k] = v.numpy()

    res = {"model": new_state_dict,
           "__author__": "convert_ts"}

    torch.save(res, output_file_name)
    print(f"Saved converted ts checkpoint to {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Models')
    parser.add_argument('input', metavar='I',
                        help='input model path')
    parser.add_argument('output', metavar='O',
                        help='output path')
    args = parser.parse_args()
    convert_ts(args.input, args.output)