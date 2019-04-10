#!/usr/bin/env python3
#--- coding: utf-8 ---

import os
import argparse 
import glob 
import pickle 
import brewer2mpl
bmap = brewer2mpl.get_map('Set1', 'Qualitative', 9)
colors = bmap.mpl_colors 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def main():

    parser = argparse.ArgumentParser(description="drawing training curves according"
            " to the loss and accurarcy log data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help="the directory storing training log")
    
    args = parser.parse_args() 

    loss_dict = {}
    acc_dict = {}

    for filename in glob.glob(os.path.join(args.log_dir, "loss_acc_*")):
        #print(filename)

        with open(filename, 'rb') as f:
            log_list = pickle.load(f)

        for entry in log_list:
            epoch = entry['epoch']
            loss = entry['loss']
            acc = entry['acc']

            if epoch in loss_dict:
                loss_dict[epoch] = min(loss_dict[epoch], loss)
            else:
                loss_dict[epoch] = loss

            if epoch in acc_dict:
                acc_dict[epoch] = max(acc_dict[epoch], acc)
            else:
                acc_dict[epoch] = acc

    epoch_list = []
    loss_list = []
    acc_list = []

    for epoch in sorted(loss_dict.keys()):
        epoch_list.append(epoch)
        loss_list.append(loss_dict[epoch])
        acc_list.append(acc_dict[epoch])

    fig, ax1 = plt.subplots() 

    line1 = ax1.plot(epoch_list, loss_list, label='Loss', color=colors[0])
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('#Epochs')

    ax2 = ax1.twinx()
    line2 = ax2.plot(epoch_list, acc_list, label='Accuracy', color=colors[1])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(40, 100)

    line1, label1 = ax1.get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    ax2.legend(line1 + line2, label1 + label2, loc='lower left',
            bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)

    plt.savefig(os.path.join(args.log_dir, 'training_curve.pdf'),
            format='pdf')

if __name__ == '__main__':
    main()
