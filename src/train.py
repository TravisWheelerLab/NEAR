# Standard library imports
import os
import random
from collections import defaultdict
from io import StringIO
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import math
import time

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import yaml

# NEAR imports
from dataloader import AlignmentDataset
from models import NEARResNet, NEARUNet

def output_examples(model, train_loader, device, output_dir, prefix, training_loss, epoch_time, num_examples=10):
    ex_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            target_matrix, query_seq, target_seq, mask_matrix, query_mask, target_mask = batch
            target_matrix, query_seq, target_seq, mask_matrix, query_mask, target_mask = target_matrix.to(device), query_seq.to(device), target_seq.to(device), mask_matrix.to(device), query_mask.to(device), target_mask.to(device)
            query_seqs = model(query_seq).detach().cpu()
            target_seqs = model(target_seq).detach().cpu()
            for i in range(target_matrix.shape[0]):
                # Increase figure size and DPI for better quality
                fig = plt.figure(figsize=(24, 16), dpi=300)
                ax1 = plt.subplot(2, 2, 1)
                ax2 = plt.subplot(2, 2, 2)
                ax3 = plt.subplot(2, 2, 3)
                ax4 = plt.subplot(2, 2, 4)
                fig.suptitle(str(os.path.join(output_dir, f'{prefix}_{ex_idx}.png')) + ": " + str(training_loss) + ", " +str(epoch_time), fontsize=16)
                ax1.imshow(target_matrix[i].detach().cpu(), cmap='PiYG', interpolation='nearest')
                ax1.set_title('Target Matrix', fontsize=14)
                ax1.tick_params(axis='both', which='major', labelsize=10)
                
                query_seq = query_seqs[i]
                target_seq = target_seqs[i]
                
                im2 = ax2.imshow(query_seq.T @ target_seq, interpolation='bicubic', cmap='PiYG')
                ax2.set_title('Query-Target Dot Product', fontsize=14)
                ax2.tick_params(axis='both', which='major', labelsize=10)
                fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                
                # Normalize query and target sequences
                query_norm = F.normalize(query_seq, p=2, dim=0)
                target_norm = F.normalize(target_seq, p=2, dim=0)
                image = query_norm.T @ target_norm
                mul = query_norm.shape[0]**0.5
                image = image * mul
                image = image - 1.5
                im3 = ax3.imshow(image, interpolation='bicubic', vmin=-(mul - 2), vmax=mul - 2, cmap='PiYG')
                ax3.set_title('Normalized Query-Target Dot Product', fontsize=14)
                ax3.tick_params(axis='both', which='major', labelsize=10)
                fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                image = query_norm.T @ target_norm
                im4 = ax4.imshow(image, interpolation='bicubic', vmin=-1, vmax=1, cmap='PiYG')
                ax4.set_title('Cosine Similarity', fontsize=14)
                ax4.tick_params(axis='both', which='major', labelsize=10)
                fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{prefix}_{ex_idx}.png'))
                plt.close('all')
                ex_idx += 1
                if ex_idx >= num_examples:
                    break

            if ex_idx >= num_examples:
                break

def train_model(model, train_loader, optimizer, num_epochs, device, norm_loss, run_name, output_dir, scheduler=None, ):

    model.to(device)
    history = defaultdict(list)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_loss_l2 = 0.0
        train_loss_cl = 0.0
        epoch_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            target_matrix, query_seq, target_seq, mask_matrix, query_mask, target_mask = batch
            target_matrix, query_seq, target_seq, mask_matrix, query_mask, target_mask = target_matrix.to(device), query_seq.to(device), target_seq.to(device), mask_matrix.to(device), query_mask.to(device), target_mask.to(device)

            query_out = model(query_seq)
            target_out = model(target_seq)

            
            #dot_matrix = torch.bmm(F.normalize(query_out, p=2, dim=-1).transpose(-1, -2), F.normalize(target_out, p=2, dim=-1))
            dot_matrix = torch.bmm(query_out.transpose(-1, -2), target_out)
            dot_matrix = dot_matrix * mask_matrix
            #softmax_matrix = torch.softmax(dot_matrix*2*math.log(query_out.shape[-1]), dim=-1)
            softmax_matrix = torch.softmax(dot_matrix, dim=-1)

            
            loss_matrix = -torch.log(softmax_matrix + 1e-7) * target_matrix

            query_out = query_out * query_mask[:,None,:]
            target_out = target_out * target_mask[:,None,:]



            L2_loss_q = query_mask * (torch.norm(query_out, p=2, dim=-2) / query_out.shape[-2]**0.5)
            L2_loss_t = target_mask * (torch.norm(target_out, p=2, dim=-2) / target_out.shape[-2]**0.5)
            
            L2_loss_q = L2_loss_q[L2_loss_q > 1.0].mean()
            L2_loss_t = L2_loss_t[L2_loss_t > 1.0].mean()

            L2_loss = None
            if L2_loss_q > 0 and L2_loss_t > 0:
                L2_loss = (L2_loss_q + L2_loss_t) / 2.0
            elif L2_loss_q > 0:
                L2_loss = L2_loss_q
            elif L2_loss_t > 0:
                L2_loss = L2_loss_t
            
            loss_cl = loss_matrix.mean()


            loss = loss_cl
            if L2_loss is not None:
                loss = loss + (norm_loss * L2_loss)
                train_loss_l2 += L2_loss.item()
            
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            train_loss_cl += loss_cl.item()
            global_step += 1

            # Log training loss every 100 batches
            if batch_idx % 100 == 0:
                print(f"{run_name}: Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.7e}, LR: {optimizer.param_groups[0]['lr']:.7e}")
            
        epoch_time = time.time() - epoch_time
        epoch_time = f'{epoch_time:.2f}'
        train_loss /= len(train_loader)
        train_loss_cl /= len(train_loader)
        train_loss_l2 /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4e} = {train_loss_cl:.7e} + {train_loss_l2:.4e}, {epoch_time} seconds')
        
        #output_examples(model, train_loader, device, output_dir, run_name + '-' + str(epoch), train_loss_cl, epoch_time)

        # Save model checkpoint for this epoch
        checkpoint_path = os.path.join(output_dir, f'{run_name}_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, checkpoint_path)

        # Log epoch-level metrics
        #writer.add_scalar('Epoch Train Loss', train_loss, epoch)


    return history


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if len(sys.argv) != 2:
    print("Usage: python train.py <config_path>")
    sys.exit(1)
config_path = sys.argv[1]
config = load_config(config_path)

optimizers = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'AdamW': torch.optim.AdamW}
models = {'UNet': NEARUNet, 'ResNet': NEARResNet}

model = models[config['model']](**config['model_args'])
optimizer = optimizers[config['optimizer']](model.parameters(), **config['optimizer_args'])
print(config)
print("--------------------------------")
print(model)

# Print the number of trainable parameters in the model
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params}')

run_name = config['run_name']
output_dir = config['output_dir']
num_epochs = config['num_epochs']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'device' in config:
    device = torch.device(config['device'])

print("--------------------------------")
print("Creating training set")
# Set up data loaders



train_dataset = AlignmentDataset(config['alignment_dir'],
                     config['query_dir'],
                      config['target_dir'],
                       config['sequence_length'],
                       config['max_offset'],
                       use_random_seq_length=config['use_random_seq_length'],
                       random_mask_rate=config['random_mask_rate'],
                       softmask=config['softmask'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)


print(len(train_dataset), "total examples. ", len(train_loader), "batches.")
# Train the model
print("Training model")
print("--------------------------------")
history = train_model(model, train_loader, optimizer, num_epochs, device, config['norm_loss'], run_name, output_dir)


# Close the TensorBoard writer
#writer.close()

