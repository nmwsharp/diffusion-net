import argparse
import os
import torch
from faust_scape_dataset import FaustScapeDataset
from model import DiffusionGeomFmaps
from utils import FrobeniusLoss, data_augmentation


def train_diffusion_geomfmaps(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, "data", "op_cache")
    model_save_path = os.path.join(base_path, f"data/saved_models/correspondence_{args.train_name}_{args.input_features}_epoch"
                                   + "_{}.pth")
    dataset_path = os.path.join(base_path, "data")

    if not os.path.exists(os.path.join(base_path, "data/saved_models/")):
        os.makedirs(os.path.join(base_path, "data/saved_models/"))

    # create dataset
    train_dataset = FaustScapeDataset(dataset_path, name=args.train_name, train=True, k_eig=args.k_eig,
                                      n_fmap=args.n_fmap, use_cache=True, op_cache_dir=op_cache_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)

    # define model
    geomfmaps = DiffusionGeomFmaps(n_feat=args.n_feat, n_fmap=args.n_fmap, lambda_=args.lambda_,
                                   input_features=args.input_features).to(device)
    optimizer = torch.optim.Adam(geomfmaps.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    criterion = FrobeniusLoss().to(device)

    # Training loop
    print("start training")
    iterations = 0
    for epoch in range(1, args.n_epochs + 1):
        geomfmaps.train()
        for i, data in enumerate(train_loader):
            shape1, shape2, C_gt = data
            *shape1, name1 = shape1
            *shape2, name2 = shape2
            shape1, shape2, C_gt = [x.to(device) for x in shape1], [x.to(device) for x in shape2], C_gt.to(device).unsqueeze(0)

            # data augmentation
            if args.input_features == "xyz":
                v1, v2 = shape1[0].clone(), shape2[0].clone()
                shape1[0] = data_augmentation(v1, rot_x=15, rot_y=15, rot_z=15, std=0.01, noise_clip=0.05,
                                              scale_min=0.9, scale_max=1.1)
                shape2[0] = data_augmentation(v2, rot_x=15, rot_y=15, rot_z=15, std=0.01, noise_clip=0.05,
                                              scale_min=0.9, scale_max=1.1)

            # do iteration
            C_pred, feat1, feat2 = geomfmaps(shape1, shape2)
            loss = criterion(C_gt, C_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            iterations += 1
            if iterations % args.log_interval == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")

        # save model
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(geomfmaps.state_dict(), model_save_path.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of GeomFmaps model.")

    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n-epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=1, help="number of epochs of training")

    parser.add_argument('--n_fmap', type=int, default=30, help='number of eigenvectors used for fmap.')
    parser.add_argument('--k_eig', type=int, default=128, help='number of eigenvectors used for diffusion.')
    parser.add_argument('--n_feat', type=int, default=128, help='number of DiffusionNet computed features')
    parser.add_argument('--lambda_', type=float, default=1e-3, help='Fmap block regularization parameter')
    parser.add_argument('--input_features', type=str, default="xyz", help='name of input features [`xyz`, `hks`]')

    parser.add_argument('--train_name', required=False, default="faust", help='name of training dataset')
    parser.add_argument('--no-cuda', action='store_true', help='Disable GPU computation')

    parser.add_argument("--checkpoint-interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--log-interval", type=int, default=10, help="interval between logging train information")

    args = parser.parse_args()
    train_diffusion_geomfmaps(args)
