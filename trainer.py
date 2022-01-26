import os
import time

import torch
import torch.distributed as torch_dist
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn import decomposition
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import losses
from utils.utils import save_checkpoint, AverageMeter


class Trainer:
    def __init__(self, args, model, optimizer, loaders, iteration, best_acc, writer_train, writer_val,
                 img_path, model_path, scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.loaders = loaders
        self.iteration = iteration
        self.best_acc = best_acc
        self.writers = {'train': writer_train, 'val': writer_val}
        self.img_path = img_path
        self.model_path = model_path
        self.scheduler = scheduler
        self.scaler = GradScaler()
        self.target = self.sizes_mask = None

    def train(self):
        # --- main loop --- #
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.run_epoch(epoch, train=True)
            accuracy_val = self.run_epoch(epoch, train=False)

            if self.args.local_rank <= 0 and not self.args.debug:
                # save checkpoint
                is_best = accuracy_val > self.best_acc
                self.best_acc = max(accuracy_val, self.best_acc)
                save_checkpoint({'epoch': epoch + 1,
                                 'net': self.args.network_feature,
                                 'state_dict': (self.model.module if hasattr(self.model, 'module') else
                                                self.model).state_dict(),
                                 'best_acc': self.best_acc,
                                 'optimizer': self.optimizer.state_dict(),
                                 'iteration': self.iteration,
                                 'scheduler': self.scheduler.state_dict()},
                                is_best, filename=os.path.join(self.model_path, f'epoch{epoch + 1}.pth.tar'),
                                keep_all=False)

    def test(self):
        if self.args.test_info == 'compute_accuracy':
            accuracies_test = self.run_epoch(epoch=None, train=False)
            if self.args.local_rank <= 0:
                print('Accuracies test:')
                print(accuracies_test)
        else:
            print(f'Test {self.args.test_info} is not implemented')

    def run_epoch(self, epoch, train=True, return_all_acc=False):
        if self.args.device == "cuda":
            torch.cuda.synchronize()
        if train:
            self.model.train()
        else:
            self.model.eval()

        avg_meters = {k: AverageMeter() for k in ['losses', 'accuracy', 'hier_accuracy', 'top1', 'top3', 'top5',
                                                  'pos_acc', 'neg_acc', 'p_norm', 'g_norm', 'batch_time', 'data_time']}

        time_last = time.time()

        loader = self.loaders['train' if train else ('val' if epoch is not None else 'test')]
        desc = f'Training epoch {epoch}' if train else (f'Evaluating epoch {epoch}' if epoch is not None else 'Testing')
        stop_total = int(len(loader) * (self.args.partial if train else 1.0))

        with tqdm(loader, desc=desc, disable=self.args.local_rank > 0, total=stop_total) as t:
            pca_features = []
            pca_labels = []
            for idx, (input_seq, labels, *indices) in enumerate(t):
                #Patch
                #p = (labels != -1)[labels == False]
                #p = labels[labels == -1]
                #if(labels.shape[0] == 1 and len(labels[labels[0, :, 0] != -1]) == 0):
                #if(labels.shape[0] == 1 and ((labels != -1)[False]).shape[0] != 0):
                if(labels.shape[0] == 1 and labels[labels == -1].shape[0] != 0):
                    continue
                if idx >= stop_total:
                    break
                # Measure data loading time
                avg_meters['data_time'].update(time.time() - time_last)
                input_seq = input_seq.to(self.args.device)
                pca_labels.append(labels.detach().numpy()) #TODO plot
                labels = labels.to(self.args.device)

                # Get sequence predictions
                with autocast(enabled=self.args.fp16):
                    with torch.set_grad_enabled(train):
                        output_model = self.model(input_seq, labels)
                        #TODO plot
                        #pca_features.append(self.model.hidden.reshape(-1).detach().numpy())
                        pca_features.append(self.model.hidden.reshape(-1).detach().cpu().numpy())
                        

                    if self.args.cross_gpu_score:
                        pred, feature_dist, sizes_pred = output_model
                        sizes_pred = sizes_pred.float().mean(0).int()

                        if self.args.parallel == 'ddp':
                            tensors_to_gather = [pred, feature_dist, labels]
                            for i, v in enumerate(tensors_to_gather):
                                tensors_to_gather[i] = gather_tensor(v)
                            pred, feature_dist, labels = tensors_to_gather

                        if self.target is None:
                            self.target, self.sizes_mask = losses.compute_mask(self.args, sizes_pred, labels.shape[0])

                        loss, *results = losses.compute_loss(self.args, feature_dist, pred, labels, self.target,
                                                             sizes_pred, self.sizes_mask, labels.shape[0])
                    else:
                        loss, results = output_model
                    losses.bookkeeping(self.args, avg_meters, results)

                del input_seq

                if train:
                    # Backward pass
                    scaled_loss = self.scaler.scale(loss.mean())
                    scaled_loss.backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                del loss

                avg_meters['batch_time'].update(time.time() - time_last)
                time_last = time.time()

                # ------------- Show information ------------ #
                postfix_kwargs = {k: v.val for k, v in avg_meters.items() if v.count > 0}
                t.set_postfix(**postfix_kwargs)

                if train and self.args.local_rank <= 0:
                    acuracy_list_train = {k: v for k, v in postfix_kwargs.items() if 'time' not in k}
                    self.iteration += 1
                    if self.iteration % self.args.print_freq == 0 and self.writers['train'] and not self.args.debug:
                        num_outer_samples = (self.iteration + 1) * self.args.batch_size * \
                                            (1 if 'Parallel' in str(type(self.model)) else self.args.step_n_gpus)
                        self.writers['train'].add_scalars('train', {**acuracy_list_train}, num_outer_samples)

            accuracy_list = {k: v.local_avg if train else v.avg for k, v in avg_meters.items()
                             if v.count > 0 and 'time' not in k}

            if not train and self.args.local_rank <= 0:
                print(f'[{epoch}/{self.args.epochs}]' +
                      ''.join([f'{k}: {", ".join([f"{v_:.04f}" for v_ in v.avg_expanded])}, '
                               for k, v in avg_meters.items() if v.count > 0]))
                if not self.args.debug: # and len(accuracy_list) > 0:
                    self.writers['val'].add_scalar('global/loss', accuracy_list['losses'], epoch)
                    self.writers['val'].add_scalars('accuracy', accuracy_list, epoch)

            #TODO Testing clustering in hyperbolic with PCA
            #if True:#train:
            mpca = False
            if train and mpca:
                pca = decomposition.PCA(n_components=2)
                f = pca.fit_transform(pca_features)
                principalDf = pd.DataFrame(data = f
                    , columns = ['principal component 1', 'principal component 2'])
                aux = np.array(pca_labels)[:,0,0,0] #action indexes
                aux = np.array(['Prepare cloth', 'Ironing', 'Let iron', 'Take Iron'])[aux]#aux.tolist()
                print(aux.shape)
                df_labels = pd.DataFrame(data = aux, columns = ['target'])
                finalDf = pd.concat([principalDf, df_labels[['target']]], axis = 1)
                plot(finalDf)

            return accuracy_list if return_all_acc else accuracy_list['accuracy']

    def get_base_model(self):
        if 'DataParallel' in str(type(self.model)):  # both the ones from apex and from torch.nn
            return self.model.module
        else:
            return self.model


def plot(finalDf):
    #fig = plt.figure(figsize = (8,8))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = ['Prepare cloth', 'Ironing', 'Let iron', 'Take Iron']
    colors = ['r', 'g', 'b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , s = 50)

    #ax.plot(finalDf)

    ax.grid()
    fig.savefig('mplot.png')
    plt.close()

def gather_tensor(v):
    if v is None:
        return None

    # list where each element is [N x H_DIM]
    gather_dest = [torch.empty_like(v) * i for i in range(torch_dist.get_world_size())]
    torch_dist.all_gather(gather_dest, v.contiguous())  # as far as i recall, this loses gradient information completely

    gather_dest[torch_dist.get_rank()] = v  # restore tensor with gradient information
    gather_dest = torch.cat(gather_dest)

    # gather_dest is now a tensor of [(N*N_GPUS) x H_DIM], as if you ran everything on one GPU, except only N samples
    # corresponding to GPU i inputs will have gradient information
    return gather_dest
