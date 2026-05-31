import os
import datetime
from collections import OrderedDict
import contextlib
from collections import deque

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir, cur_train_samples, interval)

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        self.debug_integrity_checks = getattr(settings, "debug_integrity_checks", True)
        self.debug_mode_checks = getattr(settings, "debug_mode_checks", False)
        self.debug_log_lr_each_step = getattr(settings, "debug_log_lr_each_step", True)
        self.iou_window_size = int(getattr(settings, "iou_window_size", 50))
        self.recent_iou = OrderedDict({loader.name: deque(maxlen=self.iou_window_size) for loader in self.loaders})
        if use_amp:
            self.scaler = GradScaler()

    @staticmethod
    def _parameter_checksum(model):
        sum_v = 0.0
        sq_sum_v = 0.0
        with torch.no_grad():
            for p in model.parameters():
                if p is None:
                    continue
                p_det = p.detach().float()
                sum_v += p_det.sum().item()
                sq_sum_v += (p_det * p_det).sum().item()
        return sum_v, sq_sum_v

    @staticmethod
    def _bn_buffer_checksums(model):
        checks = OrderedDict()
        with torch.no_grad():
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    if module.running_mean is not None:
                        checks[f"{module_name}.running_mean"] = module.running_mean.detach().float().sum().item()
                    if module.running_var is not None:
                        checks[f"{module_name}.running_var"] = module.running_var.detach().float().sum().item()
                    if hasattr(module, "num_batches_tracked") and module.num_batches_tracked is not None:
                        checks[f"{module_name}.num_batches_tracked"] = float(module.num_batches_tracked.detach().cpu().item())
        return checks

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)
        if self.debug_mode_checks:
            print(f"[ModeCheck] epoch={self.epoch} loader={loader.name} loader.training={loader.training} model.training={self.actor.net.training} grad_enabled={torch.is_grad_enabled()}")

        val_param_before = None
        val_bn_before = None
        if self.debug_integrity_checks and not loader.training:
            val_param_before = self._parameter_checksum(self.actor.net)
            val_bn_before = self._bn_buffer_checksums(self.actor.net)
            print(f"[ValCheck] epoch={self.epoch} pre-val param_checksum(sum={val_param_before[0]:.6e}, sq_sum={val_param_before[1]:.6e}), bn_buffers={len(val_bn_before)}")

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            inference_ctx = torch.inference_mode if not loader.training else contextlib.nullcontext
            autocast_ctx = autocast if self.use_amp else contextlib.nullcontext
            with inference_ctx():
                with autocast_ctx():
                    loss, stats = self.actor(data)

            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        grad_norm_value = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                        stats["Grad/global_norm"] = grad_norm_value
                        stats["Grad/clip_ratio"] = min(1.0, self.settings.grad_clip_norm / (grad_norm_value + 1e-12))
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                        grad_norm_value = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                        stats["Grad/global_norm"] = grad_norm_value
                        stats["Grad/clip_ratio"] = min(1.0, self.settings.grad_clip_norm / (grad_norm_value + 1e-12))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                if self.debug_integrity_checks:
                    # Validation must never update parameters through optimizer/scaler.
                    pass

            # if loader.training and self.debug_log_lr_each_step and self.lr_scheduler is not None:
            #     lr_list = self.lr_scheduler.get_last_lr()
            #     lr_str = ", ".join([f"{lr:.6e}" for lr in lr_list])
            #     print(f"[LRStep] epoch={self.epoch} loader={loader.name} iter={i} lr=[{lr_str}]")

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

        if self.debug_integrity_checks and not loader.training:
            val_param_after = self._parameter_checksum(self.actor.net)
            val_bn_after = self._bn_buffer_checksums(self.actor.net)
            param_unchanged = (val_param_before == val_param_after)
            bn_changed = [k for k, v in val_bn_after.items() if k in val_bn_before and val_bn_before[k] != v]
            print(f"[ValCheck] epoch={self.epoch} post-val param_checksum(sum={val_param_after[0]:.6e}, sq_sum={val_param_after[1]:.6e})")
            print(f"[ValCheck] epoch={self.epoch} param_unchanged={param_unchanged}")
            if bn_changed:
                print(f"[ValCheck][WARNING] BN buffers changed during val: {len(bn_changed)} tensors. Example: {bn_changed[:3]}")
            else:
                print(f"[ValCheck] BN buffers unchanged during val.")

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)
            if name == "IoU":
                if hasattr(val, "item"):
                    val = val.item()
                self.recent_iou[loader.name].append(float(val))

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)
            if len(self.recent_iou[loader.name]) > 0:
                iou_recent = sum(self.recent_iou[loader.name]) / len(self.recent_iou[loader.name])
                print_str += 'IoU@%d: %.5f  ,  ' % (self.iou_window_size, iou_recent)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
