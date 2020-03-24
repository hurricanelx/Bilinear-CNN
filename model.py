import torch
import torchvision
import torch.nn as nn

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benckmark = True


class BCNN(nn.Module):
    def __init__(self, num_class):
        super(BCNN, self).__init__()
        self.feature = torchvision.models.vgg16(pretrained=True)
        self.fc = nn.Linear(in_features=512 * 512, out_features=num_class, bias=True)

    def forward(self, x):
        N = x.shape[0]
        x = self.feature(x)
        
        x = torch.reshape(x, (N, 512, 28 * 28))
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (28 * 28)
        assert x.size() == (N, 512, 512)
        x = torch.reshape(x, (N, 512 * 512))

        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
                                          
        x = x.view(N, -1)
        out = self.fc(x)
        return out
