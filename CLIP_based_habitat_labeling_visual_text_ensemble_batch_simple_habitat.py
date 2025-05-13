# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 16:03:59 2025

@author: wenzt
"""

import os
import torch
import clip
from PIL import Image
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# 1. loading pretrained model

def load_model(path = 'C:\\document\\code\\AS4L\\pretrain_encoder\\'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(
        "ViT-L/14@336px", device=device,
        download_root=path
    )
    return model, preprocess, device


def build_class_prototypes(example_dir: str, model, preprocess, device) -> Dict[int, torch.Tensor]:
    class_feats: Dict[int, torch.Tensor] = {}
    for clsname in sorted(os.listdir(example_dir)):
        cls_path = os.path.join(example_dir, clsname)
        if not os.path.isdir(cls_path):
            continue
        feats = []
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img = Image.open(os.path.join(cls_path, fname)).convert("RGB")
                inp = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feats.append(model.encode_image(inp))
        if feats:
            cat = torch.cat(feats, dim=0)
            class_feats[int(clsname)] = cat.mean(dim=0, keepdim=True)
    return class_feats


def get_prompt_prototypes(prompts_dict: Dict[int, List[str]], model, device) -> Dict[int, torch.Tensor]:
    proto: Dict[int, torch.Tensor] = {}
    for cid, texts in prompts_dict.items():
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            txt_feats = model.encode_text(tokens)
        proto[cid] = txt_feats.mean(dim=0, keepdim=True)
    return proto


def normalize_dict(d: Dict[int, float]) -> Dict[int, float]:
    vals = torch.tensor(list(d.values()))
    norm = (vals - vals.mean()) / (vals.std() + 1e-6)
    return {k: norm[i].item() for i, k in enumerate(d.keys())}


# 2. batch prediction

def ensemble_predict_batch(
    image_paths: List[str],
    sub_protos: Dict[int, torch.Tensor],
    sub_text_protos: Dict[int, torch.Tensor],
    model, preprocess, device,
    batch_size: int = 16
) -> List[Dict[str,int]]:
    """
    batch processing
    """
    results = []
    # for i in range(0, len(image_paths), batch_size):
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch predicting", unit="batch"):
        batch_paths = image_paths[i:i+batch_size]
        imgs, valid = [], []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
                valid.append(p)
            except Exception as e:
                print(f"❌ Failed to load {p}: {e}")
        if not imgs:
            continue
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)

        # cal similarity
        def compute_sims(feats, protos):
            return {cid: (feats @ feat.T).squeeze(1) for cid, feat in protos.items()}

        sub_img_sims  = compute_sims(feats, sub_protos)
        sub_txt_sims  = compute_sims(feats, sub_text_protos)

        # fuse sim for final prediction
        B = feats.size(0)
        for idx in range(B):
            # substrate
            raw_sub_img = {cid: float(sub_img_sims[cid][idx]) for cid in sub_img_sims}
            raw_sub_txt = {cid: float(sub_txt_sims[cid][idx]) for cid in sub_txt_sims}
            norm_img    = normalize_dict(raw_sub_img)
            norm_txt    = normalize_dict(raw_sub_txt)
            fused_sub   = {cid: 0.5*norm_img[cid] + 0.5*norm_txt[cid] for cid in norm_img}
            sub_pred    = max(fused_sub.items(), key=lambda x: x[1])[0]
            
            results.append({
                "filename": os.path.basename(valid[idx]),
                "substrate_pred": sub_pred
            })
    return results



substrate_prompts = {
    0: [
        "bare sandy seabed with absolutely no visible biota",
        "clean sand bottom with no visible benthic organisms",
        "featureless sandy seafloor",
        "desert-like underwater sand with no biota"
    ],
    1: [
        "evenly scattered sessile invertebrates on the seafloor",
        "sparse but consistent distribution of benthic fauna",
        "light coverage of sponges or invertebrates",
        "uniform low-density benthic community"
    ],
    2: [
        "seafloor with patchy distribution of benthic organisms",
        "mixed coverage with dense areas and bare patches",
        "transition zone from dense to sparse benthic coverage",
        "heterogeneous distribution of sessile invertebrates"
    ],
    3: [
        "dense aggregation of sessile benthic organisms",
        "highly covered seafloor with sponges and corals",
        "rich and continuous benthic invertebrate community",
        "extensive bottom coverage by filter feeders or reef fauna"
    ]
}




# 3. main

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", default='C:\\document//code//al_planning//zeehan//imgs336', help="Image folder to predict")
    parser.add_argument("--substrate_protos", default='C:\\document//code//al_planning//zeehan//imgs//habitat_prototypes', help="Substrate prototypes folder")
    parser.add_argument("--save_csv", default='C:\\document//code//al_planning//zeehan//clip_vis_text_pred.csv', help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()

    model, preprocess, device = load_model()
    sub_protos = build_class_prototypes(args.substrate_protos, model, preprocess, device)
    sub_text   = get_prompt_prototypes(substrate_prompts, model, device)

    paths = sorted([os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    results = ensemble_predict_batch(
        paths,
        sub_protos,
        sub_text,
        model, preprocess, device,
        batch_size=args.batch_size
    )

    df = pd.DataFrame(results)
    df.to_csv(args.save_csv, index=False)
    print("✅ Batch prediction saved to", args.save_csv)
