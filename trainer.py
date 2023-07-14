# Trainers.py
import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_logger, save_checkpoint, RunningAverage
import time

class exp_log_loss:
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    
    ### ! needs raw scores from the network ! ###
    """
    def __init__(self, label_freq, device='cuda'):
        self.gamma = 0.3
        self.smooth = 1.
        self.label_freq = label_freq
        self.num_classes = len(label_freq)
        self.class_weights = torch.FloatTensor(np.power(np.full((self.num_classes), self.label_freq.sum()) / self.label_freq, 0.5)).to(device)

    def __call__(self, prediction, mask):
        # Dice loss
        dice_pred = F.softmax(prediction, dim=4)
        pred_flat = dice_pred.view(-1, self.num_classes)
        mask_flat = mask.view(-1, self.num_classes)
        intersection = (pred_flat*mask_flat).sum(dim=0)
        # numerator
        num = 2. * intersection + self.smooth
        # denominator
        denom = pred_flat.sum(dim=0) + mask_flat.sum(dim=0) + self.smooth        
        # calculate dice
        dice = num / denom
        dice_loss = torch.mean(torch.pow(torch.clamp(-torch.log(dice), min=1e-6), self.gamma))

        # XE loss
        prediction = F.log_softmax(prediction.permute(0,4,1,2,3), dim=1)  # put channels first
        mask = torch.argmax(mask, dim=4)
        xe_loss = torch.mean(torch.pow(torch.clamp(torch.nn.NLLLoss(weight=self.class_weights, reduction='none')(prediction, mask), min=1e-6), self.gamma))

        w_dice = 0.5
        w_xe = 0.5
        return (w_dice*dice_loss) + (w_xe*xe_loss)

class SegNetTrainer:
    """
    Args:
        model (SegNet): SegNet model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
        device (torch.device): device to train on
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
    """
    def __init__(self, model, optimizer, lr_scheduler, device, train_loader, val_loader, checkpoint_dir, loss_fn, max_num_epochs=1000,
                num_iterations=1, num_epoch=0, patience=10, iters_to_accumulate=4, eval_score_higher_is_better=False, logger=None):
        if logger is None:
            self.logger = get_logger('SegNetTrainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        # initialize the best_eval_score
        self.best_eval_score = float('-inf') if eval_score_higher_is_better else float('+inf')
        self.patience = patience
        self.epochs_since_improvement = 0
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.iters_to_accumulate = iters_to_accumulate
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = loss_fn

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            t = time.time()
            should_terminate = self.train(self.train_loader)
            print("Epoch trained in " + str(int(time.time()-t)) + " seconds.")
            if should_terminate:
                print("Hit termination condition...")
                break
            self.num_epoch += 1
        return self.num_iterations, self.best_eval_score

    def train(self, train_loader):
        """Trains the model for 1 epoch.
        Args:
            train_loader (torch.utils.data.DataLoader): training data loader
        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        improved = False        # for early stopping
        self.model.train()      # set the model in training mode
        for batch_idx, sample in enumerate(train_loader):
            self.optimizer.zero_grad()
            self.logger.info(f'Training iteration {self.num_iterations}. Batch {batch_idx + 1}. Epoch [{self.num_epoch + 1}/{self.max_num_epochs}]')
            image = sample['image'].type(torch.HalfTensor)
            target_mask = sample['target_mask'].type(torch.LongTensor)
            
            # send tensors to GPU
            image = image.to(self.device)
            target_mask = target_mask.to(self.device)
            
            # forward
            _, loss = self._forward_pass(image, target_mask)
            train_losses.update(loss.item(), self._batch_size(image))
            
            # simulate larger batch sizes using gradient accumulation
            loss = loss / self.iters_to_accumulate

            # Native apex mixed precision loss scaling and backward gradient computation
            self.scaler.scale(loss).backward()   
            
            # Every iters_to_accumulate, call step() to update parameters and reset gradients:
            if self.num_iterations % self.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.logger.info(f'Training stats. Loss: {train_losses.avg}')
                self._log_stats('train', train_losses.avg)
            self.num_iterations += 1

        # evaluate on validation set
        self.model.eval()
        eval_score = self.validate(self.val_loader)

        # adjust learning rate if necessary
        self.scheduler.step(eval_score)

        # log current learning rate in tensorboard
        self._log_lr()

        # remember best validation metric
        is_best = self._is_best_eval_score(eval_score)
        if(is_best):
            improved = True
        
        # save checkpoint
        self._save_checkpoint(is_best)

        # implement early stopping here
        if not improved:
            self.epochs_since_improvement += 1
        if(self.epochs_since_improvement > self.patience):  # Model has not improved for certain number of epochs
            self.logger.info(
                    f'Model not improved for {self.patience} epochs. Finishing training...')
            return True
        return False    # Continue training...
        
    def validate(self, val_loader):
        self.logger.info('Validating...')
        val_losses = RunningAverage()
        with torch.no_grad():
            for batch_idx, sample in enumerate(val_loader):
                self.logger.info(f'Validation iteration {batch_idx + 1}')
                image = sample['image'].type(torch.HalfTensor)
                target_mask = sample['target_mask'].type(torch.LongTensor)
                
                # send tensors to GPU
                image = image.to(self.device)
                target_mask = target_mask.to(self.device)
                output, loss = self._forward_pass(image, target_mask)
                val_losses.update(loss.item(), self._batch_size(image))

                # plot one val segmentation
                if (batch_idx == 0) and (self.num_epoch % 5 == 0):
                    # plot im
                    target_mask = torch.argmax(target_mask, dim=4)
                    target_mask = target_mask.cpu().numpy()[0]
                    output = torch.argmax(output, dim=4)
                    output = output.cpu().numpy()[0]
                    # axial plot
                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                    ax_slice = image.cpu().numpy().astype(float)[0, 28, :, :, 0]
                    ax0.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                    ax_slice = target_mask[28]
                    ax1.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                    ax_slice = output[28]
                    ax2.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                    self.writer.add_figure(tag='Val_seg_ax', figure=fig, global_step=self.num_epoch)
                    # sagittal plot
                    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
                    lr_slice = int(np.mean(np.where(target_mask==target_mask.max())[-1]))
                    sag_slice = image.cpu().numpy().astype(float)[0,:,:,lr_slice,0]
                    ax3.imshow(sag_slice, aspect=2.0, cmap='Greys_r')
                    sag_slice = target_mask[:,:,lr_slice]
                    ax4.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                    sag_slice = output[:,:,lr_slice]
                    ax5.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                    self.writer.add_figure(tag='Val_seg_sag', figure=fig2, global_step=self.num_epoch)
            self._log_stats('val', val_losses.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}')
            return val_losses.avg

    def _forward_pass(self, image, target_mask):
        with torch.cuda.amp.autocast():
            # forward pass
            image = image.permute(0,4,1,2,3)
            output = self.model(image)
            output = output.permute(0,2,3,4,1)                  # shuffle channels to last
            loss = self.loss_fn(output, target_mask)
            return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score
        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self._log_new_best(eval_score)
            self.best_eval_score = eval_score
            self.epochs_since_improvement = 0
        return is_best

    def _save_checkpoint(self, is_best):
        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_new_best(self, eval_score):
        self.writer.add_scalar('best_val_loss', eval_score, self.num_iterations)

    def _log_stats(self, phase, loss_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
        }
        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

    def _gen_best_val_images(self):
        self.model.load_best(self.checkpoint_dir, self.logger)
        time.sleep(5)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.val_loader):
                self.logger.info(f'Creating validation image {batch_idx+1} for best model.')
                image = sample['image'].type(torch.HalfTensor)
                target_mask = sample['target_mask'].type(torch.LongTensor)
                
                # send tensors to GPU
                image = image.to(self.device)
                target_mask = target_mask.to(self.device)
                
                # inference
                image = image.permute(0,4,1,2,3)
                output = self.model(image)
                output = output.permute(0,2,3,4,1)
                image = image.permute(0,2,3,4,1)

                # plot im
                target_mask = torch.argmax(target_mask, dim=4)
                target_mask = target_mask.cpu().numpy()[0]
                output = torch.argmax(output, dim=4)
                output = output.cpu().numpy()[0]

                # axial plot
                fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
                ax_slice = image.cpu().numpy().astype(float)[0, 24, :, :, 0]
                ax0.imshow(ax_slice, aspect=1.0, cmap='Greys_r')
                ax_slice = target_mask[24]
                ax1.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                ax_slice = output[24]
                ax2.imshow(ax_slice, aspect=1.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                self.writer.add_figure(tag='Val_seg_ax_bestModel', figure=fig, global_step=batch_idx+1)
                
                # sagittal plot
                fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(10, 3), tight_layout=True)
                lr_slice = int(np.mean(np.where(target_mask==target_mask.max())[-1]))
                sag_slice = image.cpu().numpy().astype(float)[0,:,:,lr_slice,0]
                ax3.imshow(sag_slice, aspect=2.0, cmap='Greys_r')
                sag_slice = target_mask[:,:,lr_slice]
                ax4.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                sag_slice = output[:,:,lr_slice]
                ax5.imshow(sag_slice, aspect=2.0, cmap='nipy_spectral', vmin=0, vmax=target_mask.max())
                self.writer.add_figure(tag='Val_seg_sag_bestModel', figure=fig2, global_step=batch_idx+1)

