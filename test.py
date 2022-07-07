import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter

from train import *
from model import *
from utils import fix_seed
from dataset import load_dataset_folder


parser = argparse.ArgumentParser()

parser.add_argument('-ne', '--num_epochs', default=128, type=int)
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('-s', '--seed', default=42, type=int)
parser.add_argument('-is', '--image_size', default=32, type=int)
parser.add_argument('-ic', '--in_channels', default=3, type=int)
parser.add_argument('-dh', '--dim_hidden', default=128, type=int)
parser.add_argument('-drh', '--dim_residual_hidden', default=128, type=int)
parser.add_argument('-nrl', '--num_residual_layers', default=2, type=int)
parser.add_argument('-de', '--dim_embedding', default=64, type=int)
parser.add_argument('-nes', '--num_embeddings', default=1024, type=int)
parser.add_argument('-cc', '--commitment_cost', default=0.25, type=float)
parser.add_argument('-d', '--decay', default=0.99, type=float)
parser.add_argument('-lr', '--learning_rate', default=2e-4, type=float)

args = parser.parse_args()

IN_CHANNELS = args.in_channels
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size
DIM_HIDDEN = args.dim_hidden
DIM_RESIDUAL_HIDDEN = args.dim_residual_hidden
NUM_RESIDUAL_LAYERS = args.num_residual_layers
DIM_EMBEDDING = args.dim_embedding
NUM_EMBEDDINGS = args.num_embeddings
COMMITMENT_COST = args.commitment_cost
SEED = args.seed
DECAY = args.decay
LEARNING_RATE = args.learning_rate


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


if __name__ == "__main__":
    fix_seed(SEED)
    torch.cuda.empty_cache()
    log_dir = "./logs"
    writer = SummaryWriter(log_dir)
    loss_fn = lambda lower_bound: -sum(lower_bound)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    
    # dataset_train = load_dataset_folder(path='datasets/', split='train')
    # dataset_valid = load_dataset_folder(path='datasets/', split='valid', shuffle=False)
    # for sample_train in dataset_train:
    #     sample_inputs_train = sample_train['image']
    #     break
    # train_data_variance = torch.var(torch.tensor(sample_inputs_train)).numpy()
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import torch.nn.functional as F
    import torchvision
    
    
    dataset_train = load_dataset_folder(path='datasets/', split='train')
    dataset_valid = load_dataset_folder(path='datasets/', split='valid', shuffle=False)
    
    
    def cast_and_nomrmalise_images(images):
        x = images['image']
        x = (tf.cast(x, tf.float32) / 255.0)
        images['image'] = tf.transpose(x, (2, 0, 1))
        return images
    

    # dataset_train = (
    #     train_data
    #     .map(cast_and_nomrmalise_images)
    #     .shuffle(10000)
    #     # .repeat(-1)
    #     .batch(BATCH_SIZE, drop_remainder=True)
    #     .prefetch(-1)
    # )
    # dataset_valid = (
    #     val_data
    #     .map(cast_and_nomrmalise_images)
    #     # .repeat(1)
    #     .batch(BATCH_SIZE)
    #     .prefetch(-1)
    # )
    for sample_train in dataset_train:
        sample_inputs_train = sample_train['image']
        break
    train_data_variance = np.var(np.array(sample_inputs_train))
    
    model = Model(DIM_HIDDEN, NUM_RESIDUAL_LAYERS, DIM_RESIDUAL_HIDDEN,
              NUM_EMBEDDINGS, DIM_EMBEDDING, 
              COMMITMENT_COST, DECAY)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, 
        # weight_decay=DECAY, 
        eps=0.01, amsgrad=False)
    earlystopping = EarlyStopping(path='models/')
    criterion = VQVAELoss(COMMITMENT_COST, train_data_variance)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.001)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(NUM_EPOCHS*50):
        model.to(device)
        for data in dataset_train.as_numpy_iterator():
            data = torch.tensor(data['image']).to(device)
            break
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / train_data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 4 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
            r = data_recon.detach()
            # r = torch.reshape(r, (r.shape[0], 1, 28, 28))
            grid = torchvision.utils.make_grid(r, nrow=r.shape[0])
            writer.add_image("reconstraction_image", grid, global_step=i)
    torch.save(model.to('cpu').state_dict(), "./models/"+"checkpoint.pth")
    writer.close()