import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def sequence_dropout(x_temp, y, t, kept_len):
    x_temp = x_temp[-kept_len:]
    y = y[-kept_len:]
    t = t[-kept_len:] - t[-kept_len:].max() + 1
    mask = ~torch.isnan(x_temp)

    return x_temp, y, mask, t


class Physio2019Dataset(Dataset):

    def __init__(self, df, oi_stats=None, seq_drop_prob=0, seq_drop_pct=0):
        self.df = df
        self.stat_cols = ["Age", "Gender", "adm_icu", "icu_1", "icu_2",
            "HospAdmTime"]
        self.temp_cols = [x for x in df.columns if x not in self.stat_cols and
            x not in ["pid", "ICULOS", "SepsisLabel", "subset", "beg", "end"]]
        self.include_oi = oi_stats is not None
        if self.include_oi:
            self.oi_stats = oi_stats[self.temp_cols]
        self.y_col = "SepsisLabel"
        self.t_col = "ICULOS"
        self.seq_drop_prob = seq_drop_prob
        self.seq_drop_pct = seq_drop_pct

        self.pid_df = self.df.drop_duplicates("pid", ignore_index=True)

    def __len__(self):
        return len(self.pid_df)

    def __getitem__(self, idx):
        beg = self.pid_df.beg[idx]
        end = self.pid_df.end[idx]
        cur_info = self.df.iloc[beg:end]

        x_temp = torch.from_numpy(cur_info[self.temp_cols].values).float()
        x_stat = torch.from_numpy(cur_info[self.stat_cols].values[-1:])\
            .float().squeeze(dim=0)
        mask = ~torch.isnan(x_temp)
        x_temp.masked_fill_(~mask, 0)
        x_stat[torch.isnan(x_stat)] = 0

        if self.include_oi:
            mean = torch.from_numpy(self.oi_stats.iloc[0, :].values).float()
            std = torch.from_numpy(self.oi_stats.iloc[1, :].values).float()
            oi_cumsum = (mask.cumsum(dim=0) - mean) / (std + 1e-8)
            oi_last = (mask.sum(dim=0) - mean) / (std + 1e-8)
            x_temp = torch.cat((x_temp, oi_cumsum), dim=-1)
            x_stat = torch.cat((x_stat, oi_last), dim=-1)

        y = torch.from_numpy(cur_info[self.y_col].values).long()
        t = torch.from_numpy(cur_info[self.t_col].values).float()

        length = y.size(0)

        if random.uniform(0, 1) < self.seq_drop_prob and length > 1:
            length = min(1, int((1 - self.seq_drop_pct) * length))
            x_temp, y, mask, t = sequence_dropout(x_temp, y, t,
                length)

        return x_temp, x_stat, y, t, mask, length

    @staticmethod
    def _collate_fn(samples):
        x_temp = torch.cat([sample[0] for sample in samples])
        x_stat = torch.stack([sample[1] for sample in samples])
        y_full = torch.cat([sample[2] for sample in samples])
        y_max = torch.stack([sample[2].max() for sample in samples])
        t = pad_sequence([sample[3] for sample in samples],
            batch_first=True)
        masks = pad_sequence([sample[4] for sample in samples],
            batch_first=True, padding_value=False)
        lengths = torch.tensor([sample[5] for sample in samples],
            dtype=torch.long)

        return x_temp, x_stat, y_full, y_max, t, masks, lengths

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=Physio2019Dataset._collate_fn)


class PDL1Dataset(Dataset):

    def __init__(self, df):
        self.df = df
        self.stat_cols = [x for x in self.df.columns if x.startswith("stat")]
        self.temp_cols = [x for x in self.df.columns if x.startswith("temp")]
        self.y_col = "label"
        self.t_col = "days"
        self.pid_list = self.df.pid.unique().tolist()

    def __len__(self):
        return len(self.pid_list)

    def __getitem__(self, idx):
        pid = self.pid_list[idx]
        cur_info = self.df.loc[self.df.pid == pid].reset_index(drop=True)

        x_temp = torch.from_numpy(cur_info[self.temp_cols].values).float()
        x_stat = torch.from_numpy(cur_info[self.stat_cols].values[-1:])\
            .float().squeeze(dim=0)
        mask = ~torch.isnan(x_temp)
        x_temp.masked_fill_(~mask, 0)
        x_stat[torch.isnan(x_stat)] = 0

        y = torch.from_numpy(cur_info[self.y_col].values != 2).long()
        t = torch.from_numpy(cur_info[self.t_col].values).float()

        length = y.size(0)

        return x_temp, x_stat, y, t, mask, length

    @staticmethod
    def _collate_fn(samples):
        x_temp = torch.cat([sample[0] for sample in samples])
        x_stat = torch.stack([sample[1] for sample in samples])
        y_full = torch.cat([sample[2] for sample in samples])
        y_max = torch.stack([sample[2].max() for sample in samples])
        t = pad_sequence([sample[3] for sample in samples],
            batch_first=True)
        masks = pad_sequence([sample[4] for sample in samples],
            batch_first=True, padding_value=False)
        lengths = torch.tensor([sample[5] for sample in samples],
            dtype=torch.long)

        return x_temp, x_stat, y_full, y_max, t, masks, lengths

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
            num_workers=num_workers, collate_fn=Physio2019Dataset._collate_fn)


if __name__ == "__main__":
    import pandas as pd


    info = pd.read_csv("/media/dntech/_mnt_storage/kaiming/data/simta_plus/pd_l1/pd_l1_0831.csv")
    ds = PDL1Dataset(info)
    dl = PDL1Dataset.get_dataloader(ds, 4)

    for i, sample in enumerate(dl):
        x_temp, x_stat, y_full, y_max, t, masks, lengths = sample
        print(1)
