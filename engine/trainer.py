from __future__ import annotations

import os
import yaml
import math
import contextlib
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.losses import ClassBalancedFocalLoss, ClassBalancedCELoss
from engine.utils import BuildAutocastKWargs, modalities_to_branches, format_metrics
from engine.validator import Validator


class Trainer:
    def __init__(
        self,
        cfg: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        logger,
        writer: SummaryWriter,
        world_size: int = 1,
        start_epoch: int = 0,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.world_size = world_size
        self.start_epoch = start_epoch
        self.global_step = 0
        self.optim_step: int = 0

        self.grad_clip = cfg.get("training").get("grad_clip", 0.0)
        self.accum_steps = cfg.get("training").get("accum_steps", 1)
        self.log_interval = cfg.get("training").get("log_interval", 100)

        self.autocast_kwargs = BuildAutocastKWargs(cfg, device)
        self.scaler = self._build_grad_scaler()

        counts_cfg = cfg.get("dataset").get("cls_count_dir")
        label_type = cfg.get("dataset").get("args").get("label_type")
        smoothing = cfg.get("training").get("label_smoothing", 0.0)
        beta = cfg.get("training").get("cb_beta", 0.999)
        gamma = cfg.get("training").get("focal_gamma", 3.0)
        loss_type = cfg.get("training").get("loss_type", "multiclass")
        tau = cfg.get("training").get("la_tau", 0.)
        self.criteria = self._build_criteria(loss_type, counts_cfg, label_type, smoothing, beta, gamma, tau)

        self.validator = Validator(self.device, self.autocast_kwargs)

        self.track_fusion = bool(cfg.get("training", {}).get("track_fusion_aux", False))
        self.track_classifier_aux = bool(cfg.get("training", {}).get("track_classifier_aux", False))
        model_ref = getattr(self.model, "module", self.model)
        self.mm_fusion_obj = getattr(model_ref, "mm_fusion", None)
        self.mc_fusion_objs: Dict[str, Any] = {}
        for name, br in getattr(model_ref, "branches", {}).items():
            if getattr(br, "mc_fusion", None) is not None:
                self.mc_fusion_objs[name] = br.mc_fusion

        # # track activations
        # from utils.activation_profiler import ActivationMemoryProfiler
        # from models.encoder import CNXv2Block, FMixLowRank, StageTransition
        # from models.fusion import BSX, GLRFusion
        # from models.classifier import ClassificationHead
        # from models.squeeze_excite import SoSE_X

        # self.prof = ActivationMemoryProfiler(
        #     model,
        #     include_classes=[CNXv2Block, FMixLowRank, StageTransition, BSX, GLRFusion, ClassificationHead, SoSE_X],
        #     # or filter by name: include_name_regex=r"(encoder|BSX|GLRFusion|classifier)"
        #     )

    # ------------------------- public API -----------------------------

    def train(
        self,
        epochs: int,
        ckpt_dir: str,
        validate_interval: int = 1,
        checkpoint_interval: int = 1,
    ):
        os.makedirs(ckpt_dir, exist_ok=True)

        self.total_epochs = epochs

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, epochs):
            if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            self._train_one_epoch(epoch)

            val_loss = None
            if self.val_loader is not None and (epoch + 1) % validate_interval == 0:
                metrics, val_loss = self.validate(epoch)
                self.logger.info(f"Validation @ epoch {epoch+1}: {format_metrics(metrics)}")

            save_best = False
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best = True

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            if self._is_main() and (
                ((epoch + 1) % checkpoint_interval == 0) 
                or ((epoch + 1) == epochs)
                or save_best
            ):
                print(f"Saving checkpoint for epoch {epoch+1} to {ckpt_dir}...")
                ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
                self.save_checkpoint(ckpt_path, epoch)

            # self.prof.print_summary(sort_by="saved_bytes", topk=50)

        self.close()


    def validate(self, epoch: int):
        if self.val_loader is None:
            return {}, None
        results, val_loss = self.validator(self.model, self.val_loader, self._compute_loss)
        if val_loss is not None:
            results["loss"] = val_loss
        if self._is_main():
            if val_loss is not None:
                self.writer.add_scalar("val/loss", val_loss, epoch+1)
            for group_name, heads in results.items():
                if group_name == "loss":
                    continue
                for head_name, res in heads.items():
                    for k, v in res.items():
                        tag = f"val/{group_name}_{head_name}_{k}"
                        if k == "confusion_matrix":
                            cm = torch.as_tensor(v, dtype=torch.float32)
                            cm = (cm - cm.min()) / (cm.max() - cm.min() + 1e-8)
                            cm = torch.nn.functional.interpolate(cm[None, None], size=(256, 256), mode="nearest")[0, 0]
                            self.writer.add_image(tag, cm, epoch+1, dataformats="HW")
                        elif isinstance(v, (list, tuple, np.ndarray)) and np.size(v) > 1:
                            vals = v if isinstance(v, (list, tuple)) else v.tolist()
                            for i, vi in enumerate(vals):
                                self.writer.add_scalar(f"{tag}/class_{i+1}", float(vi), epoch+1)
                        else:
                            self.writer.add_scalar(tag, float(np.array(v).squeeze()), epoch+1)

        return results, val_loss


    def save_checkpoint(self, path: str, epoch: int):
        if not self._is_main():
            return
        state = {
            "model": getattr(self.model, "module", self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": self.global_step,
        }
        torch.save(state, path)


    def resume_from(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        getattr(self.model, "module", self.model).load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.global_step = state.get("global_step", 0)
        self.start_epoch = state.get("epoch")
        self.logger.info(f"Resumed from checkpoint {path} at epoch {state.get('epoch')}")
        return state

    # ------------------------- private API -----------------------------

    def _train_one_epoch(self, epoch: int):
        
        self.model.train()

        running_loss = 0.0
        running_count = 0
        interval_loss = 0.0
        interval_count = 0
        steps = 0
        is_ddp = hasattr(self.model, "no_sync")

        for step, (batch_data, batch_labels) in enumerate(self.train_loader):
            steps += 1
            batch_data = {k: (v[0].to(self.device), v[1].to(self.device)) for k, v in batch_data.items()}
            batch_data = modalities_to_branches(batch_data)
            batch_labels = {k: v.to(self.device) for k, v in batch_labels.items()}

            ind_label, grp_label = self._prepare_label_views(batch_labels)

            with autocast(**self.autocast_kwargs):
                features, logits = self.model(batch_data, epoch=epoch)
                loss = self._compute_loss(logits, ind_label, grp_label)

            proto_targets = self._build_proto_targets(logits, ind_label, grp_label)

            with torch.no_grad():
                self.model.update_prototypes(features, proto_targets)

            mp_aux, mc_fusion_aux, mm_fusion_aux = self._get_fusion_aux()
            classifier_aux_data = self._get_classifier_aux()

            do_sync = ((step + 1) % self.accum_steps == 0)
            ctx = self.model.no_sync() if (is_ddp and not do_sync) else contextlib.nullcontext()
            with ctx:
                self.scaler.scale(loss / self.accum_steps).backward()
            if do_sync:
                self._optimizer_step()

            batch_size = batch_labels.get("group", batch_labels.get("individual")).shape[0]
            loss_val = loss.item() * batch_size
            count_val = batch_size
            if torch.distributed.is_initialized():
                tensor = torch.tensor([loss_val, count_val], device=self.device)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                loss_val, count_val = tensor.tolist()
            running_loss += loss_val
            running_count += count_val
            interval_loss += loss_val
            interval_count += count_val
            if self._is_main() and (step + 1) % 10 == 0 and interval_count > 0:
                avg_loss = interval_loss / interval_count
                self.logger.info(
                    f"Epoch {epoch+1} Step {step+1}/{len(self.train_loader)} Loss {avg_loss:.4f}"
                )
                self.writer.add_scalar("train/loss_step", avg_loss, self.global_step+1)
                for i, pg in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f"train/lr_group{i}", pg.get("lr", 0.0), self.global_step+1)
                self._maybe_log_fusion_aux(mp_aux, mc_fusion_aux, mm_fusion_aux, self.global_step+1)
                self._maybe_log_classifier_aux(classifier_aux_data, self.global_step+1)
                interval_loss = 0.0
                interval_count = 0
            self.global_step += 1

        if steps > 0 and (steps % self.accum_steps != 0):  # tail microbatch
            self._optimizer_step()

        if interval_count > 0 and self._is_main():
            avg_loss = interval_loss / interval_count
            self.logger.info(
                f"Epoch {epoch+1} Step {steps}/{len(self.train_loader)} Loss {avg_loss:.4f}"
            )
            self.writer.add_scalar("train/loss_step", avg_loss, self.global_step+1)
            self._maybe_log_fusion_aux(mp_aux, mc_fusion_aux, mm_fusion_aux, self.global_step+1)
            self._maybe_log_classifier_aux(classifier_aux_data, self.global_step+1)

        if running_count > 0 and self._is_main():
            epoch_loss = running_loss / running_count
            self.logger.info(f"Epoch {epoch+1} Training Loss {epoch_loss:.4f}")
            self.writer.add_scalar("train/loss", epoch_loss, epoch+1)


    def _prepare_label_views(self, labels: Dict[str, torch.Tensor]):
        ind_label = None
        grp_label = None
        if "individual" in labels:
            B, P, L = labels["individual"].shape
            ind_label = labels["individual"].view(B * P, L)
        if "group" in labels:
            grp_label = labels["group"]
        return ind_label, grp_label


    def _compute_loss(
        self,
        logits: Dict[str, Dict[str, torch.Tensor]],
        ind_label: Optional[torch.Tensor],
        grp_label: Optional[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = None

        if "individual" in logits and ind_label is not None:
            for idx, (name, lg) in enumerate(logits["individual"].items()):
                y = ind_label[:, idx]
                loss = self.criteria["individual"][name](lg, y)
                total_loss = loss if total_loss is None else total_loss + loss

        if "group" in logits and grp_label is not None:
            for idx, (name, lg) in enumerate(logits["group"].items()):
                y = grp_label[:, idx]
                loss = self.criteria["group"][name](lg, y)
                total_loss = loss if total_loss is None else total_loss + loss

        return total_loss


    def _build_proto_targets(
        self,
        logits: Dict[str, Dict[str, torch.Tensor]],
        ind_label: Optional[torch.Tensor],
        grp_label: Optional[torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        proto_targets: Dict[str, Dict[str, torch.Tensor]] = {"individual": {}, "group": {}}

        if "individual" in logits and ind_label is not None:
            for idx, (name, _lg) in enumerate(logits["individual"].items()):
                proto_targets["individual"][name] = ind_label[:, idx]

        if "group" in logits and grp_label is not None:
            for idx, (name, _lg) in enumerate(logits["group"].items()):
                proto_targets["group"][name] = grp_label[:, idx]

        return proto_targets


    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        step_opt = self.optim_step + 1
        if self._is_main() and self.writer is not None:
            self._maybe_log_gradients(step_opt)
        if self.grad_clip > 0:
            clip_val = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if self._is_main() and self.writer is not None:
                self._maybe_log_clip_val(clip_val, step_opt)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.optim_step += 1


    def _build_grad_scaler(self) -> GradScaler:
        enabled = bool(
            self.autocast_kwargs.get("enabled", False)
            and self.device.type == "cuda"
            and self.autocast_kwargs.get("dtype") is torch.float16
        )
        return GradScaler(enabled=enabled)


    def _build_criteria(
        self,
        loss_type: str,
        counts_cfg: Dict,
        label_type: str,
        smoothing: float,
        beta: float,
        gamma: float,
        tau: float,
    ) -> Dict[str, Dict[str, nn.Module]]:
        criteria: Dict[str, Dict[str, nn.Module]] = defaultdict(dict)

        if isinstance(counts_cfg, str):
            with open(counts_cfg, "r") as f:
                counts_cfg = yaml.safe_load(f)
        counts_cfg = counts_cfg.get(label_type)

        specs = []
        if getattr(self.model, "individual_classifier", None) is not None:
            specs.append(("individual", self.model.individual_classifier, counts_cfg.get("individual")))
        if getattr(self.model, "group_classifier", None) is not None:
            specs.append(("group", self.model.group_classifier, counts_cfg.get("group")))

        for split_name, classifier, counts_map in specs:
            for head_name in classifier.heads:
                c = torch.tensor(counts_map.get(head_name), dtype=torch.float, device=self.device)
                if loss_type == "ce":
                    criteria[split_name][head_name] = ClassBalancedCELoss(
                        c, beta, tau, smoothing).to(self.device)
                else:
                    criteria[split_name][head_name] = ClassBalancedFocalLoss(
                        c, beta, gamma, tau, smoothing, loss_type).to(self.device)
                    # criteria[split_name][head_name] = ClassBalancedFocalLoss(
                    #     c, beta, gamma, smoothing, loss_type).to(self.device)

        return criteria


    @staticmethod
    def _is_main() -> bool:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


    def close(self):
        # self.prof.clear()  
        if self.writer is not None:
            self.writer.close()


    def _get_fusion_aux(self):
        model_ref = getattr(self.model, "module", self.model)

        # multi_person (within modality-stage) fusion
        mp_aux = {}
        for br_name, br in model_ref.branches.items():
            if hasattr(br, "encoders"):
                for enc_name, enc in br.encoders.items():
                    if hasattr(enc, "cross"):
                        for i, stage in enumerate(enc.cross):
                            if hasattr(stage, "get_aux"):
                                aux = stage.get_aux()
                                if aux is not None:
                                    mp_aux[f"{br_name}/{enc_name}/cross{i}"] = aux         

        # multi-channel (within-branch) fusions
        mc_aux = {}
        for br_name, br in model_ref.branches.items():
            if hasattr(br, "mc_fusion") and hasattr(br.mc_fusion, "get_aux"):
                aux = br.mc_fusion.get_aux()
                if aux is not None:
                    mc_aux[br_name] = aux

        # multi-modal (across branches) fusion
        mm_aux = None
        if hasattr(model_ref, "mm_fusion") and hasattr(model_ref.mm_fusion, "get_aux"):
            mm_aux = model_ref.mm_fusion.get_aux()

        return mp_aux if mp_aux else None, mc_aux if mc_aux else None, mm_aux


    def _get_classifier_aux(self):
        """Probe classifier heads directly to get auxiliary data."""
        model_ref = getattr(self.model, "module", self.model)
        classifier_aux = {}
        
        if hasattr(model_ref, "individual_classifier") and model_ref.individual_classifier is not None:
            ind_classifier = model_ref.individual_classifier
            if hasattr(ind_classifier, "classifiers") and hasattr(ind_classifier, "classifier_type"):
                if ind_classifier.classifier_type == "cosine":
                    ind_aux = {}
                    for head_name in ind_classifier.heads:
                        classifier = ind_classifier.classifiers[head_name]
                        if hasattr(classifier, "get_aux"):
                            aux = classifier.get_aux()
                            if aux is not None:
                                ind_aux[head_name] = aux
                    if ind_aux:
                        classifier_aux["individual"] = ind_aux
        
        if hasattr(model_ref, "group_classifier") and model_ref.group_classifier is not None:
            grp_classifier = model_ref.group_classifier
            if hasattr(grp_classifier, "classifiers") and hasattr(grp_classifier, "classifier_type"):
                if grp_classifier.classifier_type == "cosine":
                    grp_aux = {}
                    for head_name in grp_classifier.heads:
                        classifier = grp_classifier.classifiers[head_name]
                        if hasattr(classifier, "get_aux"):
                            aux = classifier.get_aux()
                            if aux is not None:
                                grp_aux[head_name] = aux
                    if grp_aux:
                        classifier_aux["group"] = grp_aux
        
        return classifier_aux


    def _maybe_log_gradients(self, step_opt: int) -> None:
        if self.log_interval <= 0:
            return
        if step_opt % self.log_interval != 0:
            return

        def _grad_group_key(param_name: str) -> str:
            parts = param_name.split(".")
            if parts and parts[0] == "module":  # DDP
                parts = parts[1:]
            if not parts:
                return "unknown"
            if parts[0] == "branches" and len(parts) >= 2:
                return f"branches.{parts[1]}"
            return parts[0]

        stats = defaultdict(lambda: {"sum_abs": 0.0, "sum_sq": 0.0, "count": 0, "max_abs": 0.0, "grad_bufs": []})
        global_sq = 0.0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue

                g = grad.detach()
                sq = (g * g).sum().item()
                global_sq += sq

                module = _grad_group_key(name)
                s = stats[module]

                abs_g = g.abs()
                s["sum_abs"] += abs_g.sum().item()
                s["sum_sq"]  += sq
                s["count"]   += g.numel()
                s["max_abs"] = max(s["max_abs"], abs_g.max().item())
                s["grad_bufs"].append(g.view(-1).cpu())
            
        self.writer.add_scalar("grad/global_l2_norm", math.sqrt(global_sq), step_opt)

        for module, s in stats.items():
            if s["count"] == 0:
                continue
            mean_abs = s["sum_abs"] / s["count"]
            l2_norm  = math.sqrt(s["sum_sq"])

            self.writer.add_scalar(f"grad/{module}/mean_abs", mean_abs, step_opt)
            self.writer.add_scalar(f"grad/{module}/l2_norm",  l2_norm,  step_opt)
            self.writer.add_scalar(f"grad/{module}/max_abs",  s["max_abs"], step_opt)
            self.writer.add_histogram(f"grad/{module}/hist", torch.cat(s["grad_bufs"], dim=0), step_opt)


    def _maybe_log_clip_val(self, clip_val: float, step_opt: int) -> None:
        if self.log_interval <= 0:
            return
        if step_opt % self.log_interval != 0:
            return
        self.writer.add_scalar("grad/clip_norm", float(clip_val), step_opt)


    def _maybe_log_fusion_aux(self, mp_aux, mc_aux, mm_aux, step: int) -> None:
        if self.log_interval <= 0 or self.optim_step % self.log_interval != 0:
            return
        if not self.track_fusion or self.writer is None:
            return

        # --- SoSE_X (per encoder stage) ---
        if mp_aux is not None:
            for br_enc_stg, aux in mp_aux.items():
                if aux is None:
                    continue
                if "alpha" in aux:
                    self.writer.add_scalar(f"fusion/mp_{br_enc_stg}/alpha", float(aux["alpha"]), step)
                if "residual_mag" in aux:
                    self.writer.add_scalar(f"fusion/mp_{br_enc_stg}/residual_mag", float(aux["residual_mag"]), step)
                if "avail_per_person" in aux:
                    self.writer.add_scalar(f"fusion/mp_{br_enc_stg}/avail_per_person", float(aux["avail_per_person"]), step)

        # --- BSX (per branch) ---
        if mc_aux is not None:
            for br_name, aux in mc_aux.items():
                if aux is None: 
                    continue
                if "mask_cov_overall" in aux:
                    self.writer.add_scalar(f"fusion/mc_{br_name}/mask_cov_overall", float(aux["mask_cov_overall"]), step)
                if "mask_cov_per_stream" in aux:
                    v = aux["mask_cov_per_stream"].numpy()
                    for i, vi in enumerate(v):
                        self.writer.add_scalar(f"fusion/mc_{br_name}/mask_cov_stream_{i}", float(vi), step)
                for k in ["mean_norm", "gem_norm", "vec_norm"]:
                    if k in aux:
                        self.writer.add_scalar(f"fusion/mc_{br_name}/{k}", float(aux[k]), step)

        # --- GLR (cross-branch) ---
        if mm_aux is not None:
            if "alloc" in mm_aux and mm_aux["alloc"] is not None:
                alloc = mm_aux["alloc"].numpy()  # (B,M)
                alloc_mean = alloc.mean(axis=0)
                for i, mi in enumerate(alloc_mean):
                    self.writer.add_scalar(f"fusion/mm_alloc/branch_{i}", float(mi), step)
                # optional: histogram over batch
                for i in range(alloc.shape[1]):
                    self.writer.add_histogram(f"fusion/mm_alloc_hist/branch_{i}", torch.tensor(alloc[:, i]), step)

            if "gate" in mm_aux and mm_aux["gate"] is not None:
                gate = mm_aux["gate"].numpy()
                gate_mean = gate.mean(axis=0)
                for i, gi in enumerate(gate_mean):
                    self.writer.add_scalar(f"fusion/mm_gate/branch_{i}", float(gi), step)

            for k in ["pair_mag", "sum_mag", "head_mag", "beta"]:
                if k in mm_aux:
                    self.writer.add_scalar(f"fusion/mm_{k}", float(mm_aux[k]), step)


    def _maybe_log_classifier_aux(self, classifier_aux: Dict[str, Dict[str, Dict[str, torch.Tensor]]], step: int) -> None:
        if self.log_interval <= 0:
            return
        if self.optim_step % self.log_interval != 0:
            return
        if not self.track_classifier_aux or self.writer is None:
            return
        if not classifier_aux:
            return

        for split_name, split_aux in classifier_aux.items():
            for head_name, head_aux in split_aux.items():
                if head_aux is None:
                    continue
                
                if "fuse_logit_sigmoid" in head_aux:
                    fuse_sigmoid = head_aux["fuse_logit_sigmoid"]  # (K,)
                    for class_idx in range(fuse_sigmoid.shape[0]):
                        self.writer.add_scalar(
                            f"classifier/{split_name}_{head_name}_fuse_sigmoid_class_{class_idx+1}", 
                            float(fuse_sigmoid[class_idx]), 
                            step
                        )
