import torch
from torch.utils.data import Dataset
import numpy as np
from ..preprocessing.normalize import normalize_iq

class RaptorDataset(Dataset):
    def __init__(self, samples, win=4096, hop=2048, normalize="per_window"):
        self.samples=samples; self.win=win; self.hop=hop; self.normalize=normalize
        # precompute index
        self.idx=[]
        for si,s in enumerate(samples):
            T=s.iq.shape[0]
            for start in range(0, max(1,T-win+1), hop):
                self.idx.append((si,start))
        if not self.idx and samples:
            self.idx=[(0,0)]
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        si,start=self.idx[i]
        s=self.samples[si]
        iq=s.iq[start:start+self.win]  # [win,E,2]
        if iq.shape[0]<self.win:
            pad=np.zeros((self.win - iq.shape[0], iq.shape[1],2),dtype=np.float32)
            iq=np.concatenate([iq,pad],axis=0)
        iq=normalize_iq(iq, self.normalize)
        # to torch: [E, T, 2] -> we keep [T,E,2]
        item={"iq": torch.from_numpy(iq), "sample_rate": s.sample_rate, "center_freq": s.center_frequency,
              "emitter_count": s.emitter_count, "emitters": s.emitters, "site_id": s.site_id}
        return item
