from opensoundscape import Audio
import numpy as np
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opensoundscape.preprocess.actions import Action
from time import time as timer
from pytorch_metric_learning.losses import SubCenterArcFaceLoss

import sys

sys.path.append("../../src/")
from model import (
    Resnet18_Classifier,
    Resnet50_Classifier,
    ContrastiveResnet18,
    ContrastiveResnet50,
    Resnet18_Embedder,
    Resnet50_Embedder,
    HawkEarsOneModel,
    hawkears_preprocessor,
)
from preprocessor import OvenbirdPreprocessor
from dataset import PointCodeDataset, PointCodeSampler
import evaluation
import wandb
from loss import ssl_location_loss
import yaml

import bioacoustics_model_zoo as bmz

def identity(x):
    return x


print("loaded packages")
start_time = timer()

# config file path will be a command line arg, can set manually for dev purposes:
if not '.yml' in sys.argv[-1]:
    sys.argv.append(
        "/jet/home/sammlapp/song25_oven_aiid/oven_aiid/develop_and_evaluate_aiid/4_train_aiid/train_configs/base.yml"
    )
with open(sys.argv[-1], "r") as file:
    config = yaml.safe_load(file)


train_df = pd.read_csv(
    f"{config['paths']['train_clips_path']}/ovenbird_train_clips.csv"
)

# Audio clips are 5 seconds. Central 3s were originally detected as OVEN song.
# Use central section of the clip if using <5s for clip_duration
train_df["start_time"] = (5 - config["preprocessing"]["clip_duration"]) / 2
train_df["end_time"] = train_df["start_time"] + config["preprocessing"]["clip_duration"]
# compose the full path to each audio clip
train_df["file"] = train_df["clip_id"].apply(
    lambda x: f"{config['paths']['train_clips_path']}/{x}"
)
train_df = train_df.set_index(["file", "start_time", "end_time"])


# customize preprocessing according to the config
if config["preprocessing"]["use_overlay"]:
    # selection of clips that do not contain Ovenbird songs (according to HawkEars) for overlays
    overlay_df = pd.read_csv(
        f"{config['paths']['background_clips_path']}/background_clips.csv")
    overlay_df['file']=overlay_df['clip_id'].apply(lambda f: f"{config['paths']['background_clips_path']}/{f}")
    overlay_df['start_time']=0
    overlay_df['end_time']=config['preprocessing']['clip_duration']
    overlay_df = overlay_df.set_index(['file','start_time','end_time'])[[]]
else:
    overlay_df = None

if config["training"]["backbone"] == "resnet18":
    pre = hawkears_preprocessor()
pre = OvenbirdPreprocessor(overlay_df=overlay_df)
pre.pipeline.load_audio.set(
    sample_rate=32000
)  # if audio is not 32 kHz, resample to 32 kHz
if config["preprocessing"]["high_contrast"]:
    pre.pipeline.to_tensor.set(range=[-60, -20])  # increase contrast
if config["preprocessing"]["reduce_noise"]:
    pre.insert_action(
        "noise_reduce",
        after_key="trim_audio",
        action=Action(
            Audio.reduce_noise,
            is_augmentation=False,
        ),
    )


device = (
    torch.device(config["training"]["device"])
    if torch.cuda.is_available()
    else torch.device("cpu")
)


for experiment_repeat in range(config["repeats"]):
    wandb_session = wandb.init(
        project="ovenbird_aiid", entity="kitzeslab", config=config, name=config["name"]
    )

    # potentially subset the training data to a smaller size
    points = list(train_df.point_code.unique())
    if config["data"]["max_points"] is not None:
        points = np.random.choice(points, config["data"]["max_points"], replace=False)
        train_df = train_df[train_df["point_code"].apply(lambda x: x in points)]
    if config["data"]["n_train_samples"] is not None:
        replace = (
            len(train_df) < config["data"]["n_train_samples"]
        )  # only use replacement if not enough samples
        train_df = train_df.sample(config["data"]["n_train_samples"], replace=replace)
    # update points list: we may have lost points during sampling
    points = list(train_df.point_code.unique())

    # set up model, data loaders, loss, and optimizer for training
    print(f"num classes: {len(points)}")

    dataset = PointCodeDataset(train_df, pre)

    if config["training"]["loss_fn"] in (
        "cross_entropy_loss",
        "binary_cross_entropy_loss",
    ):
        strategy = "supervised"
        # dataloader selects random clips for minibatches
        train_loader = DataLoader(
            dataset=dataset,
            num_workers=config["training"]["num_workers"],
            collate_fn=identity,
            shuffle=True,
            batch_size=config["training"]["batch_size"],
        )
        if config["training"]["backbone"] == "resnet50":
            model = Resnet50_Classifier(num_classes=len(points))
        elif config["training"]["backbone"] == "hawkears":
            model = HawkEarsOneModel(num_classes=len(points))
        else:
            model = Resnet18_Classifier(num_classes=len(points))
    elif config["training"]["loss_fn"] in ("subcenter_arcface_loss",):
        strategy = "arcface"
        # dataloader selects random clips for minibatches - same as supervised
        train_loader = DataLoader(
            dataset=dataset,
            num_workers=config["training"]["num_workers"],
            collate_fn=identity,
            shuffle=True,
            batch_size=config["training"]["batch_size"],
        )
        if config["training"]["backbone"] == "resnet50":
            model = Resnet50_Embedder()
        elif config["training"]["backbone"] == "resnet18":
            model = Resnet18_Embedder()
        else:
            raise ValueError(f'unsupported backbone {config["training"]["backbone"]}')
    else:
        strategy = "contrastive"
        # PointCodeSample selects a specific number of points (recording locations) to be represented in each minibatch
        sampler = PointCodeSampler(
            train_df,
            batch_size=config["training"]["batch_size"],
            n_points_per_batch=config["training"]["n_points_per_batch"],
            n_clip_replicates=config["training"]["n_clip_replicates"],
        )
        train_loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=config["training"]["num_workers"],
            collate_fn=identity,
        )
        if config["training"]["backbone"] == "resnet50":
            model = ContrastiveResnet50()
        elif config["training"]["backbone"] == "resnet18":
            model = ContrastiveResnet18()
        else:
            raise ValueError(f'unsupported backbone {config["training"]["backbone"]}')

    print(f"training strategy: {strategy}")
    model.device = device
    model.to(device)

    loss_optimizer = None
    if config["training"]["loss_fn"] == "cross_entropy_loss":
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
    elif config["training"]["loss_fn"] == "binary_cross_entropy_loss":
        loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
    elif config["training"]["loss_fn"] == "subcenter_arcface_loss":
        embedding_size = 2048 if config["training"]["backbone"] == "resnet50" else 512
        loss_fn = SubCenterArcFaceLoss(
            embedding_size=embedding_size,
            num_classes=len(points),
            num_subcenters=config["training"]["arcface_loss_subcenters"],
        ).to(device)
        loss_optimizer = torch.optim.SGD(loss_fn.parameters(), lr=0.01)
    elif config["training"]["loss_fn"] in (
        "contrastive_point_loss",
        "ssl_contrastive_point_loss",
    ):
        # both of these strategies use contrastive loss functions
        # ssl version creates pseudo-labels iteratively, contrastive_point just uses point to force samples
        # apart and sample_id to push samples together (single sample is augmented in different ways)
        loss_fn = ssl_location_loss  # args: features,labels,clip_ids,point_ids,...
    else:
        raise ValueError(
            f"unrecognized loss_fn in config: {config['training']['loss_fn']}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["num_epochs"]
    )

    save_dir = f'{config["paths"]["save_dir"]}/{(config["name"]).replace(" ","_")}_{datetime.datetime.now().isoformat()}/'
    Path(save_dir).mkdir(exist_ok=False)

    metrics = []
    loss_hist = []
    best_ari = 0
    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # in self-supervised learning, we periodically create and update pseudo-labels
        if config["training"]["loss_fn"] == "ssl_contrastive_point":
            if (epoch) % config["training"]["clustering_interval_epochs"] == 0:
                print(f"Creating training set pseudolabels via per-point clustering")
                # dimensionality reduction is performed across all embeddings,
                # then clustering is performed on a point-by-point basis
                train_loader.dataset.clip_df["pseudo_label"], _, _ = (
                    evaluation.make_pseudolabels(
                        train_df.reset_index(),
                        model,
                        pre,
                        batch_size=config["training"]["batch_size"],
                        num_workers=config["training"]["num_workers"],
                        min_cluster_size=config["training"]["min_cluster_size"],
                        reduced_n_dimensions=config["clustering"][
                            "reduced_n_dimensions"
                        ],
                        reduction_algorithm=config["clustering"]["reduction_algorithm"],
                        cluster_grouping_col="point_code",
                        global_dim_reduction=True,
                    )
                )
                print(
                    f"N clusters: {len(train_loader.dataset.clip_df['pseudo_label'].unique())}"
                )

        with torch.set_grad_enabled(True), tqdm(total=len(train_loader)) as progressbar:

            for batch_idx, batch_samples in enumerate(train_loader):

                optimizer.zero_grad()
                model.train()
                if loss_optimizer is not None:
                    loss_optimizer.zero_grad()

                batch_data = torch.vstack([s.data[None, :, :] for s in batch_samples])
                batch_data = batch_data.to(model.device)

                # Forward pass
                # for contrastive, the outputs are the projections; for supervised, they are class logits;
                # for arcface, simply the embeddings
                embeddings, outputs = model(batch_data)

                # Loss
                if strategy == "contrastive":
                    # in self-supervised, we have pseudo-labels
                    # otherwise, we just use clip id and point ID for contrastive loss
                    if config["training"]["loss_fn"] == "ssl_contrastive_point":
                        batch_pseudolabels = torch.tensor(
                            [s.pseudo_label for s in batch_samples]
                        ).to(model.device)
                    else:
                        batch_pseudolabels = None
                    weights = config["training"]["contrastive_loss"] if 'contrastive_loss' in config['training'] else {}
                    loss = loss_fn(
                        features=outputs,
                        labels=batch_pseudolabels,
                        clip_ids=[s.clip_id for s in batch_samples],
                        point_ids=[s.point_code for s in batch_samples],
                        device=model.device,
                        **weights,
                    )
                else:  # supervised
                    # labels: numeric index positions for point codes
                    batch_labels = torch.tensor(
                        [points.index(s.point_code) for s in batch_samples]
                    )
                    if config["training"]["loss_fn"] == "binary_cross_entropy_loss":
                        # cast numeric labels to one-hot
                        batch_labels = F.one_hot(
                            batch_labels, num_classes=len(points)
                        ).float()

                    batch_labels = batch_labels.to(model.device)
                    loss = loss_fn(outputs, batch_labels)

                loss_hist.append(loss.item())
                progressbar.set_postfix(dict(loss=loss.item()))

                # Backward pass
                loss.backward()

                # Update weights using optimizer(s)
                optimizer.step()
                if loss_optimizer is not None:
                    loss_optimizer.zero_grad()

                progressbar.update(1)

        progressbar.close()
        scheduler.step()  # learning rate scheduler update

        # validation
        print("Evaluating on validation set")
        metrics_epoch = evaluation.run_eval(
            model,
            labels_dir=config["paths"]["labeled_clips_path"],
            split="val",
            pre=pre,
        )
        metrics_epoch["train_loss"] = loss_hist[-1]

        # during experiments, we discard models rather than saving weights!
        if metrics_epoch['ari']>best_ari:
            best_ari = metrics_epoch['ari']
            if "save_checkpoints" in config and config["save_checkpoints"]==True:
                
                torch.save(model.state_dict(), f"{save_dir}/best.pth")


        print(f"Validation set ARI at epoch {epoch}: {metrics_epoch['ari']}")
        metrics.extend([metrics_epoch])
        if wandb_session is not None:
            wandb_session.log(metrics_epoch)

    # save metrics, config, and loss
    metrics = pd.DataFrame(metrics).round(4)
    metrics.to_csv(f"{save_dir}/metrics.csv", index=False)
    # include some quick summary stats and metrics in the saved config
    config["results"] = {}
    config["results"]["best_epoch"] = metrics["ari"].idxmax()
    config["results"]["metrics"] = metrics.loc[
        config["results"]["best_epoch"]
    ].to_dict()
    config["results"]["train_time_hrs"] = (timer() - start_time) / 60 / 60
    config["strategy"] = strategy
    with open(f"{save_dir}/config.yml", "w") as f:
        yaml.safe_dump(config, f)
    pd.DataFrame({"loss": loss_hist}).round(6).to_csv(f"{save_dir}/loss.csv")

    print(f"finished training \nresults saved to {save_dir}")
    wandb.finish()
