import torch
import torch.nn as nn

class LocalAttentionModule(nn.Module):
    def __init__(self, channel,ratio = 16):
        super(LocalAttentionModule, self).__init__()
        self.squeeze1 = nn.AdaptiveMaxPool2d(1)
        self.squeeze2 = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()

        y1 = self.squeeze1(x).view(b, c)
        y2 = self.squeeze2(x).view(b, c)
        y = y1 + y2
        z = self.excitation(y).view(b, c, 1, 1)
        z = z.expand_as(x)
        return x * z

class SharedMLP(nn.Module):
        def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                transpose=False,

                bn=False,
                activation_fn=None
        ):
            super(SharedMLP, self).__init__()

            conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

            self.conv = conv_fn(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,

            )
            self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
            self.activation_fn = activation_fn

        def forward(self, input):
            r"""
                Forward pass of the network

                Parameters
                ----------
                input: torch.Tensor, shape (B, d_in, N, K)

                Returns
                -------
                torch.Tensor, shape (B, d_out, N, K)
            """
            x = self.conv(input)

            if self.batch_norm:
                x = self.batch_norm(x)
            if self.activation_fn:
                x = self.activation_fn(x)
            return x

class AttentivePooling(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(AttentivePooling, self).__init__()

            self.score_fn = nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.Softmax(dim=-2)
            )
            self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())
            self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channels, eps=1e-3),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1))

        def forward(self, x):
            r"""
                Forward pass

                Parameters
                ----------
                x: torch.Tensor, shape (B, d_in, N, K)

                Returns
                -------
                torch.Tensor, shape (B, d_out, N, 1)
            """
            # computing attention scores
            scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            # print(scores.shape)  torch.Size([1, 128, 1339, 1])
            # sum over the neighbors
            features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
            # print(features.shape)  #torch.Size([1, 128, 2000, 1])
            # print(x.shape)torch.Size([1, 128, 2000, 1])
            embed = self.mlp(features)
            # print(x.shape)torch.Size([1, 128, 2000, 1])
            # print(a.shape)torch.Size([1, 500, 2000, 1])
            # print(x.squeeze(3).shape)torch.Size([1, 128, 2000])

            # a=a.squeeze(3)

            # print(embed.shape)  torch.Size([1, 500, 2000, 1])
            S = torch.softmax(embed, dim=2).squeeze(3)
            # print(So.shape)
            # print(a.shape)
            # S = lip2d(So, a)#torch.Size([1, 250, 1000])
            # print(S.shape)
            out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
            """
                        [1,500,2000]
                transpose 转置矩阵 [1,2000,500,1]
             """
            # print(out.shape)  torch.Size([1, 128, 500, 1])
            return out
if __name__ == '__main__':
    x=torch.randn(1,128,2000,1)
    print(x.shape)
    z=LocalAttentionModule(128)
    x = z(x)
    print(z)
    print(z.shape)