import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 9e8


class MLPLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, norm=None, actv=None,
            drop_prob=0):
        layers = [nn.Linear(in_channels, out_channels)]
        if norm is not None:
            layers.append(norm(out_channels))
        if actv is not None:
            layers.append(actv())
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        super().__init__(*layers)


class MLP(nn.Sequential):

    def __init__(self, num_channels, num_classes, actv):
        mlp_layers = [MLPLayer(num_channels[i], num_channels[i + 1],
            nn.BatchNorm1d, actv)
            for i in range(len(num_channels) - 1)]
        mlp_layers.append(nn.Linear(num_channels[-1], num_classes))
        super().__init__(*mlp_layers)


def calculate_dt(t):
    t = t[:, None, :]
    t_repeat = t.repeat(1, t.size(-1), 1)
    dt = t_repeat - t_repeat.permute(0, 2, 1)
    dt = dt / (dt.max() + 1e-8)

    return dt


def pad_packed_sequences(packed_seq, lengths):
    # packed_seq: N x C
    # lengths: B
    b = len(lengths)
    t = lengths.max().int()
    c = packed_seq.size(-1)
    padded_seq = torch.zeros(b, t, c, dtype=torch.float,
        device=packed_seq.device)

    beg = 0
    for i in range(b):
        end = beg + lengths[i]
        padded_seq[i, :lengths[i]] = packed_seq[beg:end]
        beg = end

    return padded_seq


def pack_padded_sequences(padded_seq, lengths):
    # padded_seq: B x T x C
    # lengths: B
    b = len(lengths)
    n = lengths.sum().int()
    c = padded_seq.size(-1)
    packed_seq = torch.zeros(n, c, dtype=torch.float,
        device=padded_seq.device)

    beg = 0
    for i in range(b):
        end = beg + lengths[i]
        packed_seq[beg:end] = padded_seq[i, :lengths[i]]
        beg = end

    return packed_seq


def repeat_static_features(x_stat, lengths, num_valid_seq):
    x_stat_repeat = torch.zeros(num_valid_seq, x_stat.size(1),
        dtype=torch.float, device=x_stat.device)
    beg = 0
    for i in range(len(lengths)):
        end = beg + lengths[i]
        x_stat_repeat[beg:end] = x_stat[i]
        beg = end

    return x_stat_repeat


class SimtaPlusLayer(nn.Module):

    def __init__(self, temp_in_channels, stat_in_channels, temp_out_channels,
            stat_out_channels, drop_prob):
        super().__init__()

        self.non_linear_temp = MLPLayer(temp_in_channels + stat_in_channels,
            temp_out_channels, nn.BatchNorm1d, nn.LeakyReLU, drop_prob)
        self.non_linear_stat = MLPLayer(stat_in_channels, stat_out_channels,
            nn.BatchNorm1d, nn.LeakyReLU, drop_prob)

    def _get_self_attention(self, x):
        d = torch.tensor(x.size(-1), device=x.device)
        qk = torch.bmm(x, x.permute(0, 2, 1))
        attn = F.softmax(qk / torch.sqrt(d), dim=-1)

        return attn

    def forward(self, x_temp, x_stat, temp_attn, mask_repeat, lengths):
        # x_temp: N x C
        # x_stat: B x C
        # temp_attn: B x T x T
        # mask_repeat: B x T x T
        # lengths: N
        x_stat_repeat = repeat_static_features(x_stat, lengths,
            x_temp.size(0))

        x_temp = self.non_linear_temp(torch.cat((x_temp, x_stat_repeat),
            dim=-1))
        x_temp = pad_packed_sequences(x_temp, lengths)
        self_attn = self._get_self_attention(x_temp)
        attn = (self_attn + temp_attn).masked_fill(~mask_repeat, 0)
        x_stat = self.non_linear_stat(x_stat)
        x_temp = torch.bmm(attn, x_temp)
        x_temp = pack_padded_sequences(x_temp, lengths)

        return x_temp, x_stat


class SimtaPlus(nn.Module):

    def __init__(self, temp_in_channels, stat_in_channels,
            temp_hidden_channels, stat_hidden_channels, attn_hidden_channels,
            num_layers, num_classes, drop_prob):
        super().__init__()

        self.attn_mlp = MLP((1, attn_hidden_channels), 1, nn.LeakyReLU)
        self.non_linear_temp = MLPLayer(temp_in_channels,
            temp_hidden_channels, nn.BatchNorm1d, nn.LeakyReLU)
        self.non_linear_stat = MLPLayer(stat_in_channels,
            stat_hidden_channels, nn.BatchNorm1d, nn.LeakyReLU)
        self.num_layers = num_layers
        self.simta_plus_layers = nn.ModuleList([
            SimtaPlusLayer(temp_hidden_channels, stat_hidden_channels,
            temp_hidden_channels, stat_hidden_channels,
            drop_prob) for _ in range(num_layers)])
        last_channels = temp_hidden_channels + stat_hidden_channels
        self.classifier = nn.Linear(last_channels, num_classes)

    def _get_temporal_attention(self, dt, mask_repeat):
        b, t, _ = dt.size()
        dt = dt.reshape(-1, 1)
        attn = self.attn_mlp(dt)
        attn = attn.reshape(b, t, t)
        attn = F.softmax(attn.masked_fill(~mask_repeat, -9e8), dim=-1)

        return attn

    def forward(self, x_temp, x_stat, t, masks, lengths):
        # x_temp: N x C
        # x_stat: B x C
        # t: B x T
        # masks: B x T x C
        # lengths: B

        # calculate dt
        dt = calculate_dt(t)

        # mask reduction: B x T x C -> B x T
        masks = torch.any(masks, dim=-1)
        masks_repeat = masks[:, None, :].repeat(1, t.size(-1), 1)
        temp_attn = self._get_temporal_attention(dt, masks_repeat)
        # non-linear transform, project x_temp and x_stat to hidden dimension
        x_temp = self.non_linear_temp(x_temp)
        x_stat = self.non_linear_stat(x_stat)

        # simta-plus layers
        for i in range(self.num_layers):
            x_temp, x_stat = self.simta_plus_layers[i](x_temp, x_stat,
                temp_attn, masks_repeat, lengths)

        x_stat_repeat = repeat_static_features(x_stat, lengths,
            x_temp.size(0))

        features = torch.cat((x_temp, x_stat_repeat), dim=-1)
        output = self.classifier(features)

        # return the time-step with maximum probability
        b = x_stat.size(0)
        num_classes = output.size(-1)
        output_max = torch.zeros(b, num_classes, dtype=torch.float,
            device=output.device)
        beg = 0
        for i in range(len(lengths)):
            end = beg + lengths[i]
            output_diff = output[beg:end, 1] - output[beg:end, 0]
            output_max[i] = output[beg:end][output_diff.argmax()]
            beg = end

        return output, output_max


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    import pandas as pd
    # from torchinfo import summary

    import configs as cfg
    from dataset import Physio2019Dataset


    model = SimtaPlus(
        68,
        40,
        64,
        64,
        16,
        3,
        2,
        0.1
    )
    df = pd.read_csv(cfg.DF_PATH)
    oi_stats = pd.read_csv(cfg.OI_STATS_PATH)
    ds = Physio2019Dataset(df, oi_stats)
    dl = Physio2019Dataset.get_dataloader(ds, 128)

    for _, sample in enumerate(dl):
        x_temp, x_stat, y_full, y_max, t, masks, lengths = sample
        summary(model, input_data=(x_temp, x_stat, t, masks, lengths))
        break
