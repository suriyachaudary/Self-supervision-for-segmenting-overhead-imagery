# Resnet code taken from https://github.com/pytorch/vision/blob/v0.2.0/torchvision/models/resnet.py

import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable


def bilinear(mode = 'bilinear', scale_factor=2):
    "bilinear upsampling"
    return nn.Upsample(scale_factor=scale_factor, mode=mode)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mask = None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, average_pool_size = 7, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(average_pool_size)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class ResNet_EncoderDecoder_wbottleneck(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_EncoderDecoder_wbottleneck, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bottlleneck = nn.Conv2d(512*block.expansion, 512, kernel_size=4, padding=0, bias=False, groups=512*block.expansion)
        self.bn_bottleneck = nn.BatchNorm2d(512)

        self.deconv0 = nn.ConvTranspose2d(512, 512, kernel_size=8, stride=4, padding=2, bias=False)
        self.bn_d0 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d5 = nn.BatchNorm2d(32)
        ### output size: 64x256x256; apply regressor
        self.classifier = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True)  
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def decode(self, x):

        x = self.deconv0(x)
        x = self.bn_d0(x)
        x = self.relu(x)

        x = self.deconv1(x)
        x = self.bn_d1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn_d2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.bn_d3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.bn_d4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.bn_d5(x)
        x = self.relu(x)
        
        x = self.classifier(x)
        x = self.tanh(x)
        
        return x

    def forward(self, x):
        e = self.encode(x)
        b = self.bn_bottleneck(self.bottlleneck(e))
        d = self.decode(b)

        return d


class ResNet_EncoderDecoder(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_EncoderDecoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.deconv1 = nn.ConvTranspose2d(512*block.expansion, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_d5 = nn.BatchNorm2d(32)
        ### output size: 64x256x256; apply regressor
        self.classifier = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True)  
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def decode(self, x):

        x = self.deconv1(x)
        x = self.bn_d1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn_d2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.bn_d3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.bn_d4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.bn_d5(x)
        x = self.relu(x)
        
        x = self.classifier(x)
        x = self.tanh(x)
        
        return x

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return d


def crop(variable,tr,tc):
    r, c = variable.size()[2:]
    r1 = int(round((r - tr) / 2.))
    c1 = int(round((c - tc) / 2.))
    return variable[:,:, r1:r1+tr,c1:c1+tc]
    
class FCNify(nn.Module):
    def __init__(self, original_model, n_class=21, 
        layers_to_remove=['avgpool', 'fc7', 'bn7', 'relu7', 'fc8']):
        super(FCNify, self).__init__()
        for layers_ in layers_to_remove:        
            del(original_model._modules[layers_])
        
        self.features = copy.deepcopy(original_model)

        self.score_layer2 = nn.Conv2d(512/4, n_class, 1)
        self.score_layer3 = nn.Conv2d(1024/4, n_class, 1)
        self.score_layer4 = nn.Conv2d(2048/4, n_class, 1)
        self.upsample2 = bilinear(scale_factor = 2)
        self.upsample4 = bilinear(scale_factor = 4)
        self.upsample8 = bilinear(scale_factor = 8)

        for m in [self.score_layer2, self.score_layer3, self.score_layer4]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            
    def forward(self, x):

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        layer1 = self.features.layer1(x)
        layer2 = self.features.layer2(layer1) #downsample 2^3
        layer3 = self.features.layer3(layer2) #downsample 2^4
        layer4 = self.features.layer4(layer3) #downsample 2^5

        layer2_scores = self.score_layer2(layer2)
        layer3_scores = self.upsample2(self.score_layer3(layer3))
        layer4_scores = self.upsample4(self.score_layer4(layer4))

        layer3_scores = layer3_scores[:, :, :layer2_scores.size()[2], :layer2_scores.size()[3]]
        layer4_scores = layer4_scores[:, :, :layer2_scores.size()[2], :layer2_scores.size()[3]]

        score = layer2_scores + layer3_scores + layer4_scores
        

        return self.upsample8(score)


class FCNify_v2(nn.Module):
    def __init__(self, original_model, n_class=21, layers_to_remove=['classifier', 'tanh']):
        super(FCNify_v2, self).__init__()
        torch.cuda.manual_seed(7)
        torch.manual_seed(7)
        for layers_ in layers_to_remove:        
            del(original_model._modules[layers_])
                
        self.features = copy.deepcopy(original_model)
        self.relu = nn.ReLU(inplace=True)
        self.d1 = nn.Conv2d(512, n_class, 1)
        self.d2 = nn.Conv2d(256, n_class, 1)
        self.d3 = nn.Conv2d(128, n_class, 1)
        self.d4 = nn.Conv2d(64, n_class, 1)
        self.d5 = nn.Conv2d(32, n_class, 1)
        
        self.upsample2 = bilinear(scale_factor = 2)
        self.upsample4 = bilinear(scale_factor = 4)
        self.upsample8 = bilinear(scale_factor = 8)
        self.upsample16 = bilinear(scale_factor = 16)


        ### initialize new layers with random weights
        for m in [self.d1,self.d2,self.d3,self.d4,self.d5]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            
    def forward(self, x):
        rows = x.size()[2]
        cols = x.size()[3]
        
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        layer1 = self.features.layer1(x)
        layer2 = self.features.layer2(layer1) #downsample 2^3
        layer3 = self.features.layer3(layer2) #downsample 2^4
        layer4 = self.features.layer4(layer3) #downsample 2^5

        d1 = self.features.relu(self.features.bn_d1(self.features.deconv1(layer4)))
        score_d1 = self.upsample16(self.d1(d1))
        d1 = self.features.relu(self.features.bn_d2(self.features.deconv2(d1)))
        score_d2 = self.upsample8(self.d2(d1))
        d1 = self.features.relu(self.features.bn_d3(self.features.deconv3(d1)))
        score_d3 = self.upsample4(self.d3(d1))
        d1 = self.features.relu(self.features.bn_d4(self.features.deconv4(d1)))
        score_d4 = self.upsample2(self.d4(d1))
        d1 = self.features.relu(self.features.bn_d5(self.features.deconv5(d1)))
        score_d5 = self.d5(d1)
        score = crop(score_d1, rows, cols) + crop(score_d2, rows, cols) + crop(score_d3, rows, cols) + crop(score_d4, rows, cols) + crop(score_d5, rows, cols)

        return score



class Resnet_coach_vae(nn.Module):

    def __init__(self, block, layers, drop_ratio = 0.5):
        self.inplanes = 64
        super(Resnet_coach_vae, self).__init__()
        torch.cuda.manual_seed(7)
        torch.manual_seed(7)
        self.drop_ratio = drop_ratio
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.mu = nn.Conv2d(512, 100, kernel_size=1, bias=True)  
        self.std = nn.Conv2d(512, 100, kernel_size=1, bias=True)  

        self.pred = nn.Conv2d(100, 1, kernel_size=1, bias=True)

        self.upsample = bilinear(scale_factor = 16, mode='nearest')
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.mu and m is not self.std:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)


    def get_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        mu = self.mu(x)
        logvar = self.std(x)

        z = self.reparameterize(mu, logvar)
        d = self.pred(z)
        
        return d, mu, logvar

        
    def forward(self, x, alpha = 1, use_coach = True):
        features = None
        mu = None
        logvar = None
        if not use_coach:
            size_ = x.size()
            features = Variable(torch.rand(size_[0], 1, int(size_[2]/16), int(size_[3]/16) ).cuda())
        else:
            features, mu, logvar = self.get_feature(x) 
        
        size_ = features.size()
        features = features.view(size_[0], size_[1], size_[2]*size_[3])
        p,_ = features.topk(k = int(size_[2]*size_[3]*self.drop_ratio), dim = 2)
        partitions = p[:,:, -1]
        partitions = partitions.unsqueeze(2).expand(size_[0], size_[1], size_[2]*size_[3])
        mask = self.sigmoid(alpha*(features - partitions))


        mask = mask.view(size_)

        if not self.training:
            mask = (mask>0.5).float()

        mask = self.upsample(mask)

        return mask, mu, logvar


def resnet50_encoderdecoder(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    return model

def resnet18_encoderdecoder(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model


def resnet18_encoderdecoder_wbottleneck(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = ResNet_EncoderDecoder_wbottleneck(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model


def resnet18_coach_vae(**kwargs):
    """Constructs a ResNet-50 encoder + decoder model.
    """
    model = Resnet_coach_vae(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    return model



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, average_pool_size = 7,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], average_pool_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
