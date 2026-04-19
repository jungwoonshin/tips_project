"""Download datasets used in AgenTracer paper."""
import os
from datasets import load_dataset

BASE_DIR = "/NHNHOME/WORKSPACE/0426030030_A/jungwoonshin/tips_project/dataset"


def download_and_save(name, hf_path, hf_name=None, splits=None, **kwargs):
    save_dir = os.path.join(BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Downloading: {name} ({hf_path})")
    print(f"{'='*60}")
    try:
        ds = load_dataset(hf_path, hf_name, **kwargs)
        print(f"  Available splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            out_path = os.path.join(save_dir, f"{split_name}.jsonl")
            split_data.to_json(out_path)
            print(f"  Saved {split_name}: {len(split_data)} examples -> {out_path}")
        print(f"  Done: {name}")
    except Exception as e:
        print(f"  ERROR downloading {name}: {e}")


if __name__ == "__main__":
    # 1. GAIA
    download_and_save("gaia", "gaia-benchmark/GAIA", "2023_all")

    # 2. GSM8K
    download_and_save("gsm8k", "openai/gsm8k", "main")

    # 3. HotpotQA
    download_and_save("hotpotqa", "hotpotqa/hotpot_qa", "fullwiki")

    # 4. KodCode
    download_and_save("kodcode", "KodCode/KodCode")

    # 5. MATH
    download_and_save("math", "hendrycks/competition_math")

    # 6. MBPP+ (EvalPlus sanitized version)
    download_and_save("mbppplus", "evalplus/mbppplus")

    print("\n\nAll downloads complete!")
