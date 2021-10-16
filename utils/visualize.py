import os
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.general import calculate_confusion_matrix


def visual_count_label(train_labels, val_labels, path):
    arg_train = np.argmax(train_labels, axis=1)
    arg_val = np.argmax(val_labels, axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x=arg_train,
                  ax=ax1,
                  palette=sns.color_palette('GnBu_d', 10))
    sns.countplot(x=arg_val,
                  ax=ax2,
                  palette=sns.color_palette('YlOrRd', 10))
    
    ax1.set_title('Train Data', fontsize=16)
    ax2.set_title('Valid Data', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'count_label.png'))
    plt.clf()
    
    print('Train Data Size:', len(arg_train))
    print('Valid Data Size:', len(arg_val))
    print('---------------------------------')
    print('Total:', len(arg_train) + len(arg_val))
    
def visual_confusion_matrix(submission, path):
    conf = calculate_confusion_matrix(submission['target'], submission['pred'])
    acc = np.trace(conf) / float(np.sum(conf))
    misclass = 1 - acc
    
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(conf, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(conf.shape[0])
    plt.xticks(tick_marks, np.arange(conf.shape[0], dtype=np.int), rotation=45)
    plt.yticks(tick_marks, np.arange(conf.shape[0], dtype=np.int))
    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
    thresh = conf.max() / 1.5
    
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, '{:0.4f}'.format(conf[i, j]),
                 horizontalalignment='center',
                 color='white' if conf[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.xlabel('Predicted label accuracy={:0.4f}; missclass={:0.4f}'.format(acc, misclass))
    plt.ylabel('True label')
    plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    plt.clf()
    
def visual_f1score(precision, recall, precisions, recalls, path):
    F1_score = 2 * (precision * precision) / (recall, precision)
    plt.plot(F1_score, linestyle='--', linewidth=3., label='F1-Score')
    
    F1_scores = 2 * (recalls * precisions) / (recalls + precisions)
    for i in range(len(F1_scores[0])):
        plt.plot(F1_scores[i], label=f'Class {i} F1-Score')
    
    plt.tight_layout()
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'F1-Score.png'))
    plt.clf()

def visual_learning_rate(lr, path):
    plt.plot(lr)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.savefig(os.path.join(path, 'learning_rate.png'))
    plt.clf()

def visual_accuracy(accuracy, val_accuracy, path):
    plt.plot(accuracy, label='train')
    plt.plot(val_accuracy, label='valid')
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(path, 'accuracy.png'))
    plt.clf()
    
def visual_loss(loss, val_loss, path):
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='valid')
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.clf()
