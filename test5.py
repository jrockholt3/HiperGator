import MinkowskiEngine as ME
import torch 
import numpy as np

td,xd,yd,zd = 6,157,157,113

coords = np.zeros((td*xd*yd*zd,4))
m = 0
for i in range(td):
    for j in range(xd):
        for k in range(yd):
            for l in range(zd):
                arr = np.array([i,j,k,l])
                coords[m,:] = arr[:]
                m+=1
feats = np.ones((td*xd*yd*zd,1))
coords,feats = ME.utils.sparse_collate([coords],[feats])
print(feats)
x = ME.SparseTensor(coordinates=coords, features=feats)


k1 = (3,4,4,4) # kernel shape
s1 = (2,2,2,2) # stride 
k2 = (3,4,4,4)
s2 = (2,2,2,2) 
k3 = (2,2,2,2)
s3 = (2,2,2,2) 
k4 = (1,17,17,13)
s4 = (1,2,2,2)
in_feat = 1
D = 4
kg1= ME.KernelGenerator(
    kernel_size = k1,
    stride = s1,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg2= ME.KernelGenerator(
    kernel_size = k2,
    stride = s2,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg3= ME.KernelGenerator(
    kernel_size = k3,
    stride = s3,
    region_type=ME.RegionType.HYPER_CUBE,
    dimension=4
)
kg4= ME.KernelGenerator(
    kernel_size = k4,
    stride = s4,
    region_type=ME.RegionType.HYPER_CROSS,
    dimension=4
)
conv1 = ME.MinkowskiConvolution(in_channels=1,
                out_channels=1,
                kernel_size=k1,
                stride=s1,
                bias=True,
                kernel_generator=kg1,
                dimension=D).double()
conv2 = ME.MinkowskiConvolution(in_channels=1,
                out_channels=1,
                kernel_size=k2,
                stride=s2,
                bias=True,
                kernel_generator=kg2,
                dimension=D).double()
conv3 = ME.MinkowskiConvolution(in_channels=1,
                out_channels=1,
                kernel_size=k3,
                stride=s3,
                bias=True,
                kernel_generator=kg3,
                dimension=D).double()
conv4 = ME.MinkowskiConvolution(in_channels=1,
                out_channels=1,
                kernel_size=k4,
                stride=s4,
                bias=True,
                dimension=D).double()
maxpool = ME.MinkowskiGlobalMaxPooling()

print('input', x.coordinates.shape, x.features.shape)
x = conv1(x)
print('conv1', x.coordinates.shape, x.features.shape)
x = conv2(x)
print('conv2', x.coordinates.shape, x.features.shape)
x = conv3(x)
print('conv3', x.coordinates.shape, x.features.shape)
x = conv4(x)
print('conv4', x.coordinates.shape, x.features.shape)
x = maxpool(x)
print('after pool', x.coordinates.shape, x.features.shape)
x = x.dense()
print('dense',x[0].shape)