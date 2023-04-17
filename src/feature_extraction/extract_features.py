import os

import timm
import torch
import argparse
import pandas as pd
from FeatureExtractionDataset import FeatureExtractionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_features', required=False, default=768, type=int)
    parser.add_argument('--reduct_type', required=False, default='PCA', type=str)
    parser.add_argument('--graph_dir', required=False, default='../neo4j2raw/artgraph2recsys', type=str)
    parser.add_argument('--image_dir', required=False, default='../data_collection/images', type=str)
    parser.add_argument('--pretrain', required=False, default='True', type=bool)
    parser.add_argument('--instance', required=False, default='vit_base_patch16_224', type=str)
    parser.add_argument('--batch_size', required=False, default=128, type=int)
    return parser.parse_args()


def get_model(args):
    if args.pretrain:
        return torch.load('vit_fine_tune_style.pt')
    return timm.create_model(args.instance).to('cuda')


def main():
    # setting
    args = parse_args()
    names = pd.read_csv(f'{args.graph_dir}/mapping/artwork_entidx2name.csv',
                        names=['idx', 'name'])
    model = get_model(args)
    model.reset_classifier(num_classes=0)
    dataset = FeatureExtractionDataset(args.image_dir, data=names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # extracting features
    model.eval()
    with torch.no_grad():
        x = torch.zeros(len(dataset), 768)
        for idx, images in tqdm(enumerate(loader)):
            x[idx * args.batch_size: (idx + 1) * args.batch_size] = model(images.to('cuda'))

    # saving
    node_feat_dir = f'{args.graph_dir}/raw/node-feat/artwork'
    if not os.path.exists(node_feat_dir):
        os.makedirs(node_feat_dir)
    x_df = pd.DataFrame(x.cpu().numpy())
    x_df.to_csv(f'{node_feat_dir}/node-feat-vit-fine-tuning.csv', index=False, header=False)


if __name__ == '__main__':
    main()