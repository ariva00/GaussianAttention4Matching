import os
import time
from torch.utils.data import DataLoader
from transmatching.Data.dataset_smpl import SMPLDataset
from transmatching.Utils.utils import approximate_geodesic_distances
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn as nn
import random
import numpy
import logging
from model import EncoderPointTransfomer

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    logger = logging.getLogger(args.run_name)
    logger.info(f"training {args.run_name}")
    logger.info(f"args: {args}")
    logger.info(f"initial sigma: {args.sigma}")

# ------------------------------------------------------------------------------------------------------------------
# BEGIN SETUP  -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    set_seed(0)

    custom_layers = ()

    for i in range(6):
        if i in args.gaussian_blocks:
            custom_layers += ('g', 'f')
        else:
            custom_layers += ('a', 'f')

    # DATASET
    data_train = SMPLDataset(args.path_data, train=True)

    # DATALOADERS
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    num_points = 1000

    # INITIALIZE MODEL
    model = EncoderPointTransfomer(
        heads=args.n_heads,
        dim_head=args.dim_head,
        custom_layers=custom_layers,
        gaussian_heads=args.gaussian_heads,
        sigma=args.sigma,
    ).to(args.device)

    if args.learn_sigma:
        params = [
            { "params": list(model.linear_in.parameters()) + list(model.encoder.parameters()) + list(model.linear_out.parameters()) },
            { "params": model.gauss_attn.parameters(), "lr": args.lr * args.lr_mult}
        ]
    else:
        for p in model.gauss_attn.parameters():
            p.requires_grad = False
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.path_model, args.run_name + ".pt"), map_location=lambda storage, loc: storage))
        optimizer.load_state_dict(torch.load(os.path.join(args.path_model, "optim." + args.run_name + ".pt"), map_location=lambda storage, loc: storage))

    initial_sigma = model.gauss_attn.sigmas.clone().detach().cpu()
    print("initial sigma: ", initial_sigma)

# ------------------------------------------------------------------------------------------------------------------
# END SETUP  -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------
# BEGIN TRAINING ---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

    print("TRAINING --------------------------------------------------------------------------------------------------")
    model = model.train()
    lossmse = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(args.n_epoch):
        logger.info(f"starting epoch {epoch}/{args.n_epoch-1}")
        start = time.time()
        epoch_loss = 0
        geod_dist = None
        for item in tqdm(dataloader_train):
            if geod_dist is None:
                geod_dist = torch.tensor(approximate_geodesic_distances(item['y'][0].cpu().numpy(), item['faces'][0].cpu().numpy())).to(args.device)
            optimizer.zero_grad(set_to_none=True)

            shapes = item["x"].to(args.device)
            shape_A = shapes[:args.batch_size // 2, :, :]
            shape_B = shapes[args.batch_size // 2:, :, :]

            if args.normalize:
                shape_A = shape_A / shape_A.abs().max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).repeat_interleave(shape_A.shape[1], dim=1).repeat_interleave(shape_A.shape[2], dim=2)
                shape_B = shape_B / shape_B.abs().max(dim=1).values.max(dim=1).values.unsqueeze(1).unsqueeze(1).repeat_interleave(shape_B.shape[1], dim=1).repeat_interleave(shape_B.shape[2], dim=2)

            if args.noise:
                if torch.rand(1).item() < args.noise_p:
                    shape_A = shape_A + ((torch.randn_like(shape_A, device=shape_A.device)) * args.noise)
                if torch.rand(1).item() < args.noise_p:
                    shape_B = shape_B + ((torch.randn_like(shape_B, device=shape_B.device)) * args.noise)

            dim_A = num_points
            permidx_A = torch.randperm(dim_A)
            shape_A = shape_A[:, permidx_A, :]
            gt_A = torch.zeros_like(permidx_A)
            gt_A[permidx_A] = torch.arange(dim_A)

            dim_B = num_points
            permidx_B = torch.randperm(dim_B)
            shape_B = shape_B[:, permidx_B, :]
            gt_B = torch.zeros_like(permidx_B)
            gt_B[permidx_B] = torch.arange(dim_B)

            sep = -torch.ones(shape_A.shape[0], 1, 3).to(args.device)

            dim_B = dim_A +1
            x = torch.cat((shape_A, sep, shape_B), 1)

            
            y = model(x)
            y_shape_A = y[:, dim_B:, :] # shape_B points in shape_A space
            y_shape_B = y[:, :dim_A, :] # shape_A points in shape_B space

            if args.no_sep_loss:
                loss = ((y_shape_A[:, gt_B, :] - shape_A[:, gt_A, :]) ** 2).sum() + \
                       ((y_shape_B[:, gt_A, :] - shape_B[:, gt_B, :]) ** 2).sum()
            else:
                loss = ((y_shape_A[:, gt_B, :] - shape_A[:, gt_A, :]) ** 2).sum() + \
                       ((y_shape_B[:, gt_A, :] - shape_B[:, gt_B, :]) ** 2).sum() + \
                       lossmse(y[:, dim_A, :],sep[:, 0, :])

            if torch.isnan(loss):
                print("\nNAN LOSS\n")
                exit()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"EPOCH: {epoch} HAS FINISHED, in {time.time() - start} SECONDS! ---------------------------------------")
        print(f"LOSS: {epoch_loss} --------------------------------------------------------------------------------------")
        os.makedirs(args.path_model, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(args.path_model, args.run_name + ".pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + args.run_name + ".pt"))

        logger.info(f"ending epoch {epoch}/{args.n_epoch-1}, time {time.time() - start} seconds, loss {epoch_loss}")

        if args.save_best and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(args.path_model, "best." + args.run_name + ".pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.path_model, "optim." + "best." + args.run_name + ".pt"))
            logger.info(f"new best epoch {epoch}/{args.n_epoch-1}, loss {epoch_loss}")

    logger.info(f"initial sigma: {initial_sigma}")
    logger.info(f"final sigma: {model.gauss_attn.sigmas.clone().detach().cpu()}")
    logger.info(f"training {args.run_name} has finished")

    print("initial sigma: ", initial_sigma)
    print("final sigma: ", model.gauss_attn.sigmas.clone().detach().cpu())


# ------------------------------------------------------------------------------------------------------------------
# END TRAINING -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def cross_heads_loss(attn:torch.Tensor):
    return (-(attn - attn.roll(1, 1)).abs()).exp().mean(dim=2).sum()

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--run_name", default="custom_trained_model", help="name of the run, determines the name of the saved model")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=5000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")

    parser.add_argument("--path_data", default="dataset/", help="path to dir containing the dataset")
    parser.add_argument("--path_model", default="./models", help="path to dir where the model will be saved")

    parser.add_argument("--resume", default=False, action="store_true", help="resume training from a saved model, the model is determined by run_name")

    parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads (Including Gaussian Heads)")
    parser.add_argument("--dim_head", type=int, default=64, help="dimension of the attention heads")

    parser.add_argument("--gaussian_heads", type=int, default=0, help="number of gaussian attention heads")
    parser.add_argument("--sigma", type=float, default=[], nargs="*", help="initial sigma for the gaussian attention heads")
    parser.add_argument("--no_sep_loss", default=False, action="store_true", help="do not use additional loss term on the separator")
    parser.add_argument("--learn_sigma", default=False, action="store_true", help="learn the sigma of the gaussian attention heads")
    parser.add_argument("--lr_mult", type=float, default=1.0, help="learning rate multiplier for the sigma parameters")

    parser.add_argument("--device", default="auto", help="device to use for training, auto will use cuda if available, mps if available, else cpu")


    parser.add_argument("--log_file", default="train.log", help="file to log the training process")

    parser.add_argument("--gaussian_blocks", type=int, default=list(range(6)), nargs="*", help="blocks to use gaussian attention in, the default is in all blocks")

    parser.add_argument("--save_best", default=False, action="store_true", help="save the model with the best training loss")

    parser.add_argument("--normalize", default=False, action="store_true", help="normalize the input shapes to the range [-1, 1]")
    parser.add_argument("--noise", type=float, default=0.0, help="add noise to the input shapes")
    parser.add_argument("--noise_p", type=float, default=0.5, help="probability of adding noise to a shape")


    args, _ = parser.parse_known_args()

    if args.gaussian_heads == 0:
        args.gaussian_heads = False
    elif len(args.sigma) != args.gaussian_heads:
        while len(args.sigma) < args.gaussian_heads:
            if args.learn_sigma:
                args.sigma.append(torch.rand(1).item())
            elif len(args.sigma) > 0:
                args.sigma.append(args.sigma[-1] * 2)
            else:
                args.sigma.append(0.05)
        args.sigma = args.sigma[:args.gaussian_heads]

    if args.noise == 0:
        args.noise = False

    if args.device == "auto":
        args.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    main(args)
