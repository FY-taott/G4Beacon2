#! /usr/bin python
# -*- coding: utf-8 -*-
# Author: Tiantong Tao, Zhuofan Zhang
# Update date: 2024/11/14

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def onehot_encoder(samples):
    res = np.zeros([samples.shape[0], samples.shape[1], 4], dtype=np.float32)
    for idx, sample in enumerate(samples):
        for jdx, base in enumerate(sample):
            res[idx][jdx][int(base)] = 1
    return res


class g4SeqEnv:
    def __init__(self,
                 vg4Seq: str = None,
                 ug4Seq: str = None,
                 vg4ATAC: str = None,
                 ug4ATAC: str = None,
                 vg4ATACFd: str = None,
                 ug4ATACFd: str = None,
                 vg4BS: str = None,
                 ug4BS: str = None,
                 normalization: bool = False,
                 **kwformat_input):
                 
        r'''
            Take feature-file-name(s) as input, load and preprocess
            to construct feature[pandas]/label[np.array] format objects.

            Note: if use **kwformat_input param, the former input param(s) will be ignored
                  except the normalization setting. Usage example of this param is in param_tuning_cv.py.
        '''
        if kwformat_input:
            vg4Seq = kwformat_input['vg4seq']
            ug4Seq = kwformat_input['ug4seq']
            vg4ATAC = kwformat_input['vg4atac']
            ug4ATAC = kwformat_input['ug4atac']
            vg4BS = kwformat_input['vg4bs']
            ug4BS = kwformat_input['ug4bs']
            vg4ATACFd = kwformat_input['vg4atacFd']  # First-diff of vg4atac
            ug4ATACFd = kwformat_input['ug4atacFd']  # First-diff of ug4atac

        pSampleNums = 0
        nSampleNums = 0

        if vg4Seq and ug4Seq:
            vg4suffix = str.lower(vg4Seq[-3:])
            ug4suffix = str.lower(ug4Seq[-3:])

            if vg4suffix == 'csv':
                vg4seqFeatures = pd.read_csv(vg4Seq, dtype='a', header=None)  # .astype(np.float32)
            elif vg4suffix == 'npy':  # Embedded-format was not actually used here. This component currently serves no immediate purpose but is reserved for future interface updates.
                vg4seqFeatures = np.load(vg4Seq)
                vg4seqFeatures = pd.DataFrame(np.reshape(vg4seqFeatures, (vg4seqFeatures.shape[0], -1)))  # mat -> vec

            if ug4suffix == 'csv':
                ug4seqFeatures = pd.read_csv(ug4Seq, dtype='a', header=None)  # .astype(np.float32)
            elif ug4suffix == 'npy':  # Embedded-format was not actually used here. This component currently serves no immediate purpose but is reserved for future interface updates.
                ug4seqFeatures = np.load(ug4Seq)
                ug4seqFeatures = pd.DataFrame(np.reshape(ug4seqFeatures, (ug4seqFeatures.shape[0], -1)))  # mat -> vec
            pSampleNums = vg4seqFeatures.shape[0]
            nSampleNums = ug4seqFeatures.shape[0]
            seqFeatures = pd.concat([vg4seqFeatures, ug4seqFeatures], ignore_index=True)
        elif vg4Seq and ug4Seq is None:
            seqFeatures = pd.read_csv(vg4Seq, dtype='a', header=None)
            pSampleNums = seqFeatures.shape[0]
        else:
            seqFeatures = None

        if vg4ATAC and ug4ATAC:
            vg4atacFeatures = pd.read_csv(vg4ATAC, dtype='a', header=None).astype(np.float32)
            ug4atacFeatures = pd.read_csv(ug4ATAC, dtype='a', header=None).astype(np.float32)
            pSampleNums = vg4atacFeatures.shape[0]
            nSampleNums = ug4atacFeatures.shape[0]
            atacFeatures = pd.concat([vg4atacFeatures, ug4atacFeatures], ignore_index=True)
            if normalization:
                atacFeatures = pd.DataFrame(normalize(atacFeatures, 'l2'))
        elif vg4ATAC and ug4ATAC is None:
            atacFeatures = pd.read_csv(vg4ATAC, dtype='a', header=None)
            pSampleNums = atacFeatures.shape[0]
        else:
            atacFeatures = None

        if vg4BS and ug4BS:
            vg4bsFeatures = pd.read_csv(vg4BS, dtype='a', header=None)
            ug4bsFeatures = pd.read_csv(ug4BS, dtype='a', header=None)
            pSampleNums = vg4bsFeatures.shape[0]
            nSampleNums = ug4bsFeatures.shape[0]
            bsFeatures = pd.concat([vg4bsFeatures, ug4bsFeatures], ignore_index=True)
            if normalization:
                bsFeatures = pd.DataFrame(normalize(bsFeatures, 'l2'))
        else:
            bsFeatures = None

        if vg4ATACFd and ug4ATACFd:
            vg4atacFdFeatures = pd.read_csv(vg4ATACFd, dtype='a', header=None)
            ug4atacFdFeatures = pd.read_csv(ug4ATACFd, dtype='a', header=None)
            pSampleNums = vg4atacFdFeatures.shape[0]
            nSampleNums = ug4atacFdFeatures.shape[0]
            atacFdFeatures = pd.concat([
                vg4atacFdFeatures, ug4atacFdFeatures],
                ignore_index=True
            )
            if normalization:
                atacFdFeatures = pd.DataFrame(normalize(atacFdFeatures, 'l2'))
        else:
            atacFdFeatures = None

        featureList = [seqFeatures, atacFeatures, bsFeatures, atacFdFeatures]
        self.Features = None
        for feature in featureList:
            if feature is not None:
                if self.Features is not None:
                    self.Features = pd.concat([self.Features, feature], axis=1, ignore_index=True)
                else:
                    self.Features = feature


        self.Labels = np.array([1 for i in range(pSampleNums)] + [0 for i in range(nSampleNums)])

    def __len__(self):
        return len(self.Labels)

    def __getitem__(self, idx):
        # return (self.seqFeatures.iloc[idx], self.envFeatures.iloc[idx]), self.Labels[idx]
        return self.Features.iloc[idx], self.Labels[idx]
