import torch.nn as nn
from torchvision import models
from HAM import ResBlock_HAM
import torch


class resnet18_HAM(nn.Module):
    def __init__(self, class_nums):
        super(resnet18_HAM, self).__init__()
        resnet18 = models.resnet18(pretrained=True)  # 得到resnet18的模型 并且拿到预训练好的参数
        model_list = list(resnet18.children())
        self.model_pre = nn.Sequential(*model_list[:3])  # 拿到resnet18一开始的conv-bn-relu
        self.ham1 = ResBlock_HAM(64)  # 第一个HAM注意力模块
        self.model_mid = nn.Sequential(*model_list[4:8])  # 拿到resnet18中间部分的resnetBlock
        self.ham2 = ResBlock_HAM(512)  # 第二个HAM注意力模块
        self.model_tail = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # 自适应池化
        self.fc = nn.Sequential(
            nn.Linear(512, 2 * 512 + 1),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2 * 512 + 1, 2 * class_nums + 1),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2 * class_nums + 1, class_nums)
        )

    def forward(self, images):
        x = self.model_pre(images)
        x = self.ham1(x)
        x = self.model_mid(x)
        x = self.ham2(x)
        x = self.model_tail(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    images = torch.randn([3, 3, 224, 224])
    model = resnet18_HAM(class_nums=100)
    out = model(images)
    print(out.shape)

