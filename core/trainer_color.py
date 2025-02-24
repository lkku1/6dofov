import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss, CalculateLoss
from core.dataset import TrainDataset
import torch.nn.functional as F
import cv2
from FlowFormer.core.utils import flow_viz
# from model.recurrent_flow_completion import RecurrentFlowCompleteNet


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_frames = config['train_data_loader']['num_frames']

        # torch.cuda.set_device(config['device'])
        # setup data set and data loader
        self.train_dataset = TrainDataset(config['train_data_loader'])
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=config['world_size'], rank=config['global_rank'])


        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.cuda(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # self.fix_flow_complete = RecurrentFlowCompleteNet(
        #     '/mnt/lustre/sczhou/VQGANs/CodeMOVI/experiments_model/recurrent_flow_completion_v5_train_flowcomp_v5/gen_760000.pth')
        # for p in self.fix_flow_complete.parameters():
        #     p.requires_grad = False
        # self.fix_flow_complete.to(self.config['device'])
        # self.fix_flow_complete.eval()

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator(in_channel=5, out_channel=3)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.cuda(self.config['device'])

        # self.interp_mode = self.config['model']['interp_mode']
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        # print(self.config['local_rank'])
        if config['distributed']:
            self.netG = DDP(self.netG, device_ids=[self.config['local_rank']], broadcast_buffers=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD, device_ids=[self.config['local_rank']], broadcast_buffers=True)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f'Params {name} will not be optimized.')

        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
        ]
        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))


        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']:
                self.scheD = MultiStepRestartLR(
                    self.optimD,
                    milestones=scheduler_opt['milestones'],
                    gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            if not self.config['model']['no_dis']:
                self.scheD = CosineAnnealingRestartLR(
                    self.optimD,
                    periods=scheduler_opt['periods'],
                    restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        if not self.config['model']['no_dis']:
            self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])

            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])

            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found.'
                      'An initialized model will be used.')


    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')


    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']

        for frames, depths, con_mask, occ_mask, fore_flow, back_flow, name in self.train_loader:

            self.iteration += 1
            frames, depths, fore_flow, back_flow, con_mask, occ_mask = frames.to(device), depths.to(device), fore_flow.to(device), back_flow.to(device), con_mask.to(device), occ_mask.to(device)

            b, t, c, h, w = frames.size()
            masked_color = torch.cat([frames * con_mask, depths * (con_mask + occ_mask), 1 - con_mask], dim=2)

            gt_flows_bi = torch.cat([fore_flow, back_flow], dim=2)
            # ---- complete flow ----
            # pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, 1-con_masks)
            # pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, 1-con_masks)
            pred_flows_bi = gt_flows_bi

            pred_imgs = self.netG(masked_color, pred_flows_bi, 1 - con_mask)
            pred_imgs = torch.tanh(pred_imgs)
            pred_imgs = pred_imgs.view(b, -1, 3, h, w)
            comp_imgs = frames * con_mask + pred_imgs * occ_mask

            if self.config['global_rank'] == 0 and self.iteration % 500 == 0:
                aa = (comp_imgs[0, 2, :3, :, :].permute(1, 2, 0).detach().cpu().numpy() + 1)/2 *255
                cv2.imwrite("1.png", aa)
                dd = ((pred_imgs * occ_mask)[0, 2, :3, :, :].permute(1, 2, 0).detach().cpu().numpy()+1)/2 *255
                cv2.imwrite("2.png", (dd))
                ee = ((frames * occ_mask)[0, 2, :3, :, :].permute(1, 2, 0).detach().cpu().numpy()+1)/2 * 255
                cv2.imwrite("3.png", (ee))

            gen_loss = 0
            dis_loss = 0
            # optimize net_g
            if not self.config['model']['no_dis']:
                for p in self.netD.parameters():
                    p.requires_grad = False

            self.optimG.zero_grad()

            hole_loss = self.l1_loss(pred_imgs * occ_mask, frames * occ_mask)
            hole_loss = hole_loss / torch.mean(occ_mask) * self.config['losses']['valid_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * con_mask, frames * con_mask)
            valid_loss = valid_loss / torch.mean(con_mask) * self.config['losses']['hole_weight']
            gen_loss += valid_loss

            # gan loss
            if not self.config['model']['no_dis']:
                # generator adversarial loss

                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss * self.config['losses']['adversarial_weight']

                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())
            gen_loss.backward()
            self.optimG.step()

            if not self.config['model']['no_dis']:
                # optimize net_d
                for p in self.netD.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()

                # discriminator adversarial loss
                real_clip = self.netD(frames * (con_mask+occ_mask))
                fake_clip = self.netD(comp_imgs.detach()*(con_mask+occ_mask))
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
                dis_loss.backward()
                self.optimD.step()
            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))
                else:
                    pbar.set_description((
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break
