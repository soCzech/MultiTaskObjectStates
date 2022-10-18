import os
import cv2
import time
import tqdm
import torch
import random
import socket
import argparse
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import lookforthechange
from method.dataset import ChangeItVideoDataset, identity_collate, DistributedDropFreeSampler
from method.model import ClipClassifier
from method.utils import get_cosine_schedule_with_warmup, AverageMeter, select_correct_classes, JointMeter

cv2.setNumThreads(1)  # do not spawn multiple threads for augmentation (ffmpeg then raises an error)


def main(args):
    ngpus_per_node = torch.cuda.device_count()
    node_count = int(os.environ.get("SLURM_NPROCS", "1"))
    node_rank = int(os.environ.get("SLURM_PROCID", "0"))
    job_id = os.environ.get("SLURM_JOBID", "0")

    if node_count == 1:  # for PBS/PMI clusters
        node_count = int(os.environ.get("PMI_SIZE", "1"))
        node_rank = int(os.environ.get("PMI_RANK", "0"))
        job_id = os.environ.get("PBS_JOBID", "".join([str(random.randint(0, 9)) for _ in range(5)]))

    dist_url = "file://{}.{}".format(os.path.realpath("distfile"), job_id)
    print(f"Hi from node {socket.gethostname()} ({node_rank}/{node_count} with {ngpus_per_node} GPUs)!", flush=True)

    mp.spawn(main_worker, nprocs=ngpus_per_node, args=({
        "ngpus_per_node": ngpus_per_node,
        "node_count": node_count,
        "node_rank": node_rank,
        "dist_url": dist_url,
        "job_id": job_id
    }, args))


def main_worker(local_rank, cluster_args, args):
    world_size = cluster_args["node_count"] * cluster_args["ngpus_per_node"]
    global_rank = cluster_args["node_rank"] * cluster_args["ngpus_per_node"] + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=cluster_args["dist_url"],
        world_size=world_size,
        rank=global_rank,
    )

    if global_rank == 0:
        for k, v in sorted(vars(args).items(), key=lambda x: x[0]):
            print(f"# {k}: {v}")
        print(f"# effective_batch_size: {world_size * args.local_batch_size}", flush=True)

    ###############
    # DATASET
    ###############
    train_ds = ChangeItVideoDataset(
        video_roots=args.video_roots, annotation_root=os.path.join(args.dataset_root, "annotations"),
        file_mode="unannotated", noise_adapt_weight_root=None if args.ignore_video_weight else os.path.join(args.dataset_root, "videos"),
        noise_adapt_weight_threshold_file=None if args.ignore_video_weight else os.path.join(args.dataset_root, "categories.csv"), augment=args.augment
    )
    test_ds = ChangeItVideoDataset(
        video_roots=args.video_roots, annotation_root=os.path.join(args.dataset_root, "annotations"),
        file_mode="annotated", noise_adapt_weight_root=None if args.ignore_video_weight else os.path.join(args.dataset_root, "videos"),
        noise_adapt_weight_threshold_file=None if args.ignore_video_weight else os.path.join(args.dataset_root, "categories.csv"), augment=False
    )

    if global_rank == 0:
        print(train_ds, test_ds, sep="\n", flush=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, shuffle=True, drop_last=True) if world_size > 1 else None
    train_ds_iter = torch.utils.data.DataLoader(
        train_ds, batch_size=args.local_batch_size, shuffle=world_size == 1, drop_last=True, num_workers=2,
        pin_memory=True, sampler=train_sampler, collate_fn=identity_collate)

    test_sampler = DistributedDropFreeSampler(test_ds, shuffle=False) if world_size > 1 else None
    test_ds_iter = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=2,
        pin_memory=True, sampler=test_sampler, collate_fn=identity_collate)

    ###############
    # MODEL
    ###############
    weights = torch.jit.load(args.clip_weights, map_location="cpu").state_dict()
    model = ClipClassifier(hidden_mlp_layers=[4096],
                           params=weights,
                           n_classes=train_ds.n_classes,
                           train_backbone=args.train_backbone)
    assert model.backbone.input_resolution == 224

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model_parallel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    ###############
    # OPTIMIZER
    ###############
    head_params = model_parallel.module.heads.parameters()
    optim_head = torch.optim.SGD(head_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler_head = get_cosine_schedule_with_warmup(optim_head, 5 * len(train_ds_iter), len(train_ds_iter) * args.n_epochs)

    if args.train_backbone:
        backbone_params = model_parallel.module.backbone.parameters()
        optim_backbone = torch.optim.AdamW(backbone_params, lr=args.lr_backbone, weight_decay=args.weight_decay_backbone)
        scheduler_backbone = get_cosine_schedule_with_warmup(optim_backbone, 5 * len(train_ds_iter), len(train_ds_iter) * args.n_epochs)

    ###############
    # TRAINING
    ###############
    n_frames_per_gt = args.n_frames_per_gt
    kappa_dist = 60

    loss_metric = AverageMeter()
    loss_norm_metric = AverageMeter()
    unsup_state_loss_metric = AverageMeter()
    unsup_action_loss_metric = AverageMeter()

    for epoch in range(1, args.n_epochs + 1):
        if world_size > 1: train_sampler.set_epoch(epoch)
        loss_metric.reset()
        loss_norm_metric.reset()
        unsup_state_loss_metric.reset()
        unsup_action_loss_metric.reset()

        iterator = tqdm.tqdm(train_ds_iter) if global_rank == 0 else train_ds_iter
        for batch in iterator: # id, class, video, annotation/None, weight

            optim_head.zero_grad()
            if args.train_backbone: optim_backbone.zero_grad()

            # COMPUTE GT FOR ALL VIDEOS IN BATCH
            batch_for_training = []
            for _, class_, inputs, _, weight in batch:
                classes = torch.LongTensor([class_])

                # PREDICT
                with torch.no_grad():
                    predictions = []
                    for i in range(0, len(inputs), 256):
                        predictions += [model(inputs[i:i + 256].cuda(local_rank))]
                    predictions = {
                        "state": torch.cat([p["state"] for p in predictions], dim=0),
                        "action": torch.cat([p["action"] for p in predictions], dim=0)
                    }

                    st_probs = select_correct_classes(
                        torch.softmax(predictions["state"].unsqueeze(0), -1), classes, n_classes=train_ds.n_classes)
                    ac_probs = select_correct_classes(
                        torch.softmax(predictions["action"].unsqueeze(0), -1), classes, n_classes=train_ds.n_classes + 1)

                # COMPUTE GROUND TRUTH
                indices = lookforthechange.optimal_state_change_indices(
                    st_probs, ac_probs, lens=torch.tensor([st_probs.shape[1]], dtype=torch.int32, device=st_probs.device))
                indices = indices.view(1, 3).cpu()  # [S1idx, S2idx, ACidx]

                positives = indices.repeat(n_frames_per_gt, 1) + \
                            torch.arange(-(n_frames_per_gt // 2), (n_frames_per_gt // 2) + 1, 1, device=indices.device).unsqueeze_(1)
                indices_extended = torch.cat([
                    positives.transpose(1, 0).reshape(-1), positives[:, 2] - kappa_dist, positives[:, 2] + kappa_dist
                ], 0).clamp_(0, len(inputs) - 1)
                # [ S1idx - 1,      S1idx,  S1idx + 1,
                #   S2idx - 1,      S2idx,  S2idx + 1,
                #   ACidx - 1,      ACidx,  ACidx + 1,
                #  ACidx - 61, ACidx - 60, ACidx - 59,
                #  ACidx + 59, ACidx + 60, ACidx + 61]

                bg_class_index = train_ds.n_classes
                action_targets = torch.LongTensor([bg_class_index] * n_frames_per_gt * 2 +
                                                  [class_] * n_frames_per_gt +
                                                  [bg_class_index] * n_frames_per_gt * 2)
                # [  BG,  BG,  BG,
                #    BG,  BG,  BG,
                #   CLS, CLS, CLS,
                #    BG,  BG,  BG,
                #    BG,  BG,  BG]

                bg_class_index = train_ds.n_classes * 2
                state_targets = torch.LongTensor([class_ * 2 + 0] * n_frames_per_gt +
                                                 [class_ * 2 + 1] * n_frames_per_gt +
                                                 [bg_class_index] * n_frames_per_gt +
                                                 [bg_class_index] * n_frames_per_gt * 2)
                state_target_mask = torch.FloatTensor([1.] * n_frames_per_gt * 3 + [0.] * n_frames_per_gt * 2)
                # [ CS1, CS1, CS1,
                #   CS2, CS2, CS2,
                #    BG,  BG,  BG,
                #     *,   *,   *,
                #     *,   *,   *]

                batch_for_training.append((
                    inputs[indices_extended], action_targets, state_targets, state_target_mask, weight))

            # FORWARD + BACKWARD PASS
            predictions = model_parallel(torch.cat([x[0] for x in batch_for_training], 0).cuda(local_rank))

            if batch_for_training[0][4] is None:
                video_loss_weight = torch.FloatTensor([1. for _ in batch_for_training]).view(-1, 1).cuda(local_rank)
            else:
                video_loss_weight = torch.FloatTensor([x[4] for x in batch_for_training]) * (-1 / 0.001)
                video_loss_weight = 1 / (1 + torch.exp(video_loss_weight))
                video_loss_weight = video_loss_weight.view(-1, 1).cuda(local_rank)

            state_gt = torch.cat([x[2] for x in batch_for_training], 0).cuda(local_rank)
            state_gt_mask = torch.cat([x[3] for x in batch_for_training], 0).cuda(local_rank)
            action_gt = torch.cat([x[1] for x in batch_for_training], 0).cuda(local_rank)

            state_loss = F.cross_entropy(predictions["state"], state_gt, reduction="none") * state_gt_mask
            state_loss = state_loss.view(-1, n_frames_per_gt * 5) * video_loss_weight
            action_loss = F.cross_entropy(predictions["action"], action_gt, reduction="none")
            action_loss = action_loss.view(-1, n_frames_per_gt * 5) * video_loss_weight

            state_loss = torch.sum(state_loss)
            action_loss = 0.2 * torch.sum(action_loss)
            loss = state_loss + action_loss

            # DistributedDataParallel does gradient averaging, i.e. loss is x-times smaller than in Look for the Change.
            # When training with frozen backbone, make it somewhat equivalent to the Look for the Change setup.
            if not args.train_backbone:
                loss = loss * world_size
            loss.backward()

            optim_head.step()
            scheduler_head.step()
            if args.train_backbone:
                optim_backbone.step()
                scheduler_backbone.step()

            loss_metric.update(loss.item(), len(batch_for_training))
            unsup_state_loss_metric.update(state_loss.item(), len(batch_for_training))
            unsup_action_loss_metric.update(action_loss.item(), len(batch_for_training))

        ###############
        # VALIDATION
        ###############
        joint_meter = JointMeter(train_ds.n_classes)
        for batch in test_ds_iter:
            _, class_, inputs, annot, _ = batch[0]
            classes = torch.LongTensor([class_])

            with torch.no_grad():
                predictions = model(inputs.cuda(local_rank))
                st_probs = select_correct_classes(
                    torch.softmax(predictions["state"].unsqueeze(0), -1), classes, n_classes=train_ds.n_classes)
                ac_probs = select_correct_classes(
                    torch.softmax(predictions["action"].unsqueeze(0), -1), classes, n_classes=train_ds.n_classes + 1)

            joint_meter.log(ac_probs[0, :, 0].cpu().numpy(), st_probs[0].cpu().numpy(), annot, category=class_)

        vallog_fn = "{}.{}".format(os.path.realpath("vallog"), cluster_args["job_id"])
        joint_meter.dump(vallog_fn, global_rank)
        dist.barrier()

        if global_rank == 0:
            time.sleep(10)
            joint_meter.load(vallog_fn)
            dir_ = f'logs/{cluster_args["job_id"]}'
            os.makedirs(dir_, exist_ok=True)
            torch.save({"state_dict": model_parallel.state_dict(), "args": model.args, "n_classes": model.n_classes,
                        "hidden_mlp_layers": model.hidden_mlp_layers}, f"{dir_}/epoch{epoch:03d}.pth")

            print(f"Epoch {epoch} ("
                  f"T loss: {loss_metric.value:.3f}, "
                  f"T lr: {scheduler_head.get_last_lr()[0]:.6f}, "
                  f"T grad norm: {loss_norm_metric.value:.1f}, "
                  f"T unsup state loss: {unsup_state_loss_metric.value:.3f}, "
                  f"T unsup action loss: {unsup_action_loss_metric.value:.3f}, "
                  f"V state acc: {joint_meter.acc:.1f}%, "
                  f"V state prec: {joint_meter.sp:.1f}%, "
                  f"V state joint prec: {joint_meter.jsp:.1f}%, "
                  f"V action prec: {joint_meter.ap:.1f}%, "
                  f"V action joint prec: {joint_meter.jap:.1f}%)", flush=True)

            print("> {:20} {:>6} {:>6} {:>6} {:>6} {:>6}".format("CATEGORY", "SAcc", "SP", "JtSP", "AP", "JtAP"))
            print("\n".join([
                "> {:20}{:6.1f}%{:6.1f}%{:6.1f}%{:6.1f}%{:6.1f}%".format(cls_name, *joint_meter[train_ds.classes[cls_name]])
                for cls_name in sorted(train_ds.classes.keys())
            ]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_roots", type=str, nargs="+", default=["./videos"])
    parser.add_argument("--dataset_root", type=str, default="./ChangeIt")
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--lr_backbone", default=0.00001, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--weight_decay_backbone", type=float, default=0.)
    parser.add_argument("--train_backbone", action="store_true")
    parser.add_argument("--n_frames_per_gt", type=int, default=3)
    parser.add_argument("--local_batch_size", type=int, default=2)
    parser.add_argument("--clip_weights", type=str, default="./weights/ViT-L-14.pt")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--ignore_video_weight", action="store_true")
    main(parser.parse_args())
