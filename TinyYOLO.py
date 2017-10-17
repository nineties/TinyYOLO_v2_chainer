import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

def darknetConv2D(in_channel, out_channel, bn=True):
    if(bn):
        return Chain(
            c  = L.Convolution2D(in_channel, out_channel, ksize=3, pad=1, nobias=True),
            n  = L.BatchNormalization(out_channel, use_beta=False, eps=0.000001),
            b  = L.Bias(shape=[out_channel,]),
        )
    else:
        return Chain(
            c  = L.Convolution2D(in_channel,out_channel, ksize=1, pad=0,nobias=True),
            b  = L.Bias(shape=[out_channel,]),
        )

# Convolution -> ReLU -> Pooling
def CRP(c, h, stride=2, pad=0, pooling=True):
    # convolution -> leakyReLU -> MaxPooling
    h = c.b(c.n(c.c(h)))
    h = F.leaky_relu(h,slope=0.1)
    if pooling:
        h = F.max_pooling_2d(h, ksize=2, stride=stride, pad=pad)
    return h

class TinyYOLO(Chain):
    def __init__(self):
        super(TinyYOLO, self).__init__(
            c1 = darknetConv2D(3, 16),
            c2 = darknetConv2D(None, 32),
            c3 = darknetConv2D(None, 64),
            c4 = darknetConv2D(None, 128),
            c5 = darknetConv2D(None, 256),
            c6 = darknetConv2D(None, 512),
            c7 = darknetConv2D(None, 1024),
            c8 = darknetConv2D(None, 1024),
            c9 = darknetConv2D(None, 125, bn=False)
        )
    def __call__(self,x):
       return self.predict(x)

    def predict(self, x):
        h = CRP(self.c1, x)
        h = CRP(self.c2, h)
        h = CRP(self.c3, h)
        h = CRP(self.c4, h)
        h = CRP(self.c5, h)
        h = CRP(self.c6, h, stride=1, pad=1)
        h = h[:,:,1:14,1:14]
        h = CRP(self.c7, h, pooling=False)
        h = CRP(self.c8, h, pooling=False)
        h = self.c9.b(self.c9.c(h)) # no leaky relu, no BN
        return h

    def loadCoef(self,filename):
        print("loading",filename)
        file = open(filename, "rb")
        dat=np.fromfile(file,dtype=np.float32)[4:] # skip header(4xint)

        layers=[[3, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024], [1024, 1024]]

        offset=0
        for i, l in enumerate(layers):
            in_ch = l[0]
            out_ch = l[1]

            layer = getattr(self, 'c%d' % (i+1))
            layer.b.b.data = dat[offset: offset+out_ch] # Bias.b
            offset += out_ch
            layer.n.gamma.data = dat[offset:offset+out_ch] # BatchNormalization.gamma
            offset += out_ch
            layer.n.avg_mean = dat[offset:offset+out_ch] # BatchNormalization.avg_mean
            offset +=out_ch
            layer.n.avg_var = dat[offset:offset+out_ch] # BatchNormalization.avg_var
            offset +=out_ch
            layer.c.W.data = dat[offset:offset+(out_ch*in_ch*9)].reshape(out_ch, in_ch, 3, 3)    # Convolution2D.W
            offset += out_ch*in_ch*9

        # load last convolution weight(BiasとConvolution2Dのみロードする)
        in_ch = 1024
        out_ch = 125
        self.c9.b.b.data = dat[offset:offset+out_ch]
        offset += out_ch
        self.c9.c.W.data = dat[offset:offset+out_ch*in_ch*1].reshape(out_ch, in_ch, 1, 1)
        offset += out_ch*in_ch*1

        print('done')

if __name__ == '__main__':
    with chainer.using_config('train', False):
        c=TinyYOLO()
        im=np.zeros((1, 3, 416, 416),dtype=np.float32) # ネットワークの入出力設定がNoneでも初回forward時にshape決まるので、とりあえず意味なく1回forwardする
        c.predict(im)

        c.loadCoef("tiny-yolo-voc.weights") # パラメータ代入
        serializers.save_npz('TinyYOLO_v2.model', c)
