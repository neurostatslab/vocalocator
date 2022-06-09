import torch


class GerbilizerDenseNet(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerDenseNet, self).__init__()

        if CONFIG["POOLING"] == "AVG":
            self.pooling = torch.nn.AvgPool1d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
        elif CONFIG["POOLING"] == "MAX":
            self.pooling = torch.nn.MaxPool1d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
        else:
            raise ValueError("Did not recognize POOLING config.")

        # Initial number of audio channels.
        N = CONFIG["NUM_MICROPHONES"]

        self.f_convs = torch.nn.ModuleList([])
        self.g_convs = torch.nn.ModuleList([])
        self.norm_layers = torch.nn.ModuleList([])

        for i in range(12):
            n = CONFIG["CONV_NUM_CHANNELS"][i]
            fs = CONFIG["CONV_FILTER_SIZES"][i]
            d = CONFIG[f"CONV_DILATIONS"][i]
            self.f_convs.append(torch.nn.Conv1d(
                N, n, fs, stride=2, padding=((fs * d - 1) // 2), dilation=d
            ))
            self.g_convs.append(torch.nn.Conv1d(
                N, n, fs, stride=2, padding=((fs * d - 1) // 2), dilation=d
            ))
            if CONFIG["USE_BATCH_NORM"]:
                self.norm_layers.append(torch.nn.BatchNorm1d(N + n))
            else:
                self.norm_layers.append(torch.nn.Identity())
            N = N + n

        # Final pooling layer, which takes a weighted average
        # over the time axis.
        self.final_pooling = torch.nn.Conv1d(
            N, N, kernel_size=10, groups=N, padding=0
        )

        # Final linear layer to reduce the number of channels.
        self.coord_readout = torch.nn.Linear(
            N, 2
        )
        """
        self.x_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        """

        # Initialize weights
        # Input/output scaling moved to dataloader
        # self.f_convs[0].weight.data.mul_(CONFIG["INPUT_SCALE_FACTOR"])
        # self.g_convs[0].weight.data.mul_(CONFIG["INPUT_SCALE_FACTOR"])
        # self.x_coord_readout.weight.data.mul_(CONFIG["OUTPUT_SCALE_FACTOR"])
        # self.y_coord_readout.weight.data.mul_(CONFIG["OUTPUT_SCALE_FACTOR"])

    def forward(self, x):

        for fc, gc, bnrm in zip(
                self.f_convs, self.g_convs, self.norm_layers
            ):
            h = torch.tanh(fc(x)) * torch.sigmoid(gc(x))
            xp = self.pooling(x)
            x = bnrm(torch.cat((xp, h), dim=1))

        x_final = torch.squeeze(self.final_pooling(x), dim=-1)
        # px = self.x_coord_readout(x_final)
        # py = self.y_coord_readout(x_final)
        return self.coord_readout(x_final)
        # return torch.stack((px, py), dim=-1)
