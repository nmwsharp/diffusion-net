import argparse
import os
from tqdm import tqdm
import torch
from faust_scape_dataset import FaustScapeDataset
from model import DiffusionGeomFmaps


def eval_diffusion_geomfmaps(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, "data", "op_cache")
    model_load_path = os.path.join(base_path, f"data/saved_models/{args.model_name}.pth")
    prediction_save_path = os.path.join(base_path, f"data/predictions/{args.predictions_name}.pt")
    dataset_path = os.path.join(base_path, "data")

    if not os.path.exists(os.path.join(base_path, "data/predictions/")):
        os.makedirs(os.path.join(base_path, "data/predictions/"))

    # create dataset
    eval_dataset = FaustScapeDataset(dataset_path, name=args.eval_name, train=False, k_eig=args.k_eig,
                                     n_fmap=args.n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=None, shuffle=False)

    # define model
    geomfmaps = DiffusionGeomFmaps(n_feat=args.n_feat, n_fmap=args.n_fmap, lambda_=args.lambda_,
                                   input_features=args.input_features).to(device)
    geomfmaps.load_state_dict(torch.load(model_load_path))
    geomfmaps.eval()

    to_save_list = []

    for i, data in tqdm(enumerate(eval_loader)):
        shape1, shape2, C_gt = data
        *shape1, name1 = shape1
        *shape2, name2 = shape2
        shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(device) for x in shape2], C_gt.to(device).unsqueeze(0)

        # do iteration
        C_pred, _, _ = geomfmaps(shape1, shape2)

        to_save_list.append((name1, name2, C_pred.detach().cpu().squeeze(0), C_gt.detach().cpu().squeeze(0),
                             shape1[6].detach().cpu(), shape2[6].detach().cpu(),
                             shape1[10].detach().cpu(), shape2[10].detach().cpu()))

    torch.save(to_save_list, prediction_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of GeomFmaps model.")

    parser.add_argument('--n_fmap', type=int, default=30, help='number of eigenvectors used for fmap.')
    parser.add_argument('--k_eig', type=int, default=128, help='number of eigenvectors used for diffusion.')
    parser.add_argument('--n_feat', type=int, default=128, help='number of DiffusionNet computed features')
    parser.add_argument('--lambda_', type=float, default=1e-3, help='Fmap block regularization parameter')
    parser.add_argument('--input_features', type=str, default="xyz", help='name of input features [`xyz`, `hks`]')

    parser.add_argument('--eval_name', required=False, default="faust", help='name of training dataset')
    parser.add_argument('--no-cuda', action='store_true', help='Disable GPU computation')

    parser.add_argument("--model_name", type=str, help="model to be used for prediction (should exist in `saved_models` folder")
    parser.add_argument("--predictions_name", type=str, help="name of the prediction file")

    args = parser.parse_args()
    eval_diffusion_geomfmaps(args)
