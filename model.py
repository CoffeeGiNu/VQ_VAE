import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):

    def __init__(self, in_channels, dim_hidden, num_residual_hidden) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_residual_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=num_residual_hidden,
            out_channels=dim_hidden,
            kernel_size=1,
            stride=1,
        )
    

    def forward(self, x):
        y = F.relu(self.conv_1(x))
        y = F.relu(self.conv_2(y) + x)
        return y


class ResidualStack(nn.Module):

    def __init__(self, in_channels, dim_hidden, 
        num_residual_layers, num_residual_hiddens) -> None:
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([
            Residual(in_channels, dim_hidden, num_residual_hiddens)
            for _ in range(self._num_residual_layers)
        ])
    
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, dim_hidden, 
        num_residual_layers, dim_residual_hidden, name=None) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._dim_hidden = dim_hidden
        self._num_residual_layers = num_residual_layers
        self._dim_residual_hidden = dim_residual_hidden
    
        self._enc_1 = nn.Conv2d(
            in_channels, 
            dim_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1                           
        )
        self._enc_2 = nn.Conv2d(
            dim_hidden //2, 
            dim_hidden,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._enc_3 = nn.Conv2d(
            dim_hidden, dim_hidden,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._residual_stack = ResidualStack(
            dim_hidden,
            dim_hidden,
            num_residual_layers,
            dim_residual_hidden,
        )
    
    def forward(self, inputs):
        h = torch.relu(self._enc_1(inputs))
        h = torch.relu(self._enc_2(h))
        h = self._enc_3(h)
        return self._residual_stack(h)


class VectorQuantizer(nn.Module):

    def __init__(self, dim_embedding, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._dim_embedding = dim_embedding
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._w = nn.Embedding(self._num_embeddings, self._dim_embedding)
        self._w.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
     
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.size()
        input_flattened = inputs.view(-1, self._dim_embedding)
        distances = (
            torch.sum(input_flattened ** 2, dim=1, keepdim=True) 
            - 2 * torch.matmul(input_flattened, self._w.weight.t())
            + torch.sum(self._w.weight ** 2, dim=1))
        encoding_indices = torch.argmax(-distances, 1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._w.weight)
        quantized = quantized.view(input_shape) 
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return {
            'distances': distances,
            'quantize': quantized,
            'vq_loss': vq_loss, 
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'perplexity': perplexity,
        }


class Decoder(nn.Module):
    def __init__(self, in_channels, dim_hidden,  num_residual_layers, 
        dim_residual_hidden, name=None):
        super(Decoder, self).__init__()
        self._in_channels = in_channels
        self._dim_hidden = dim_hidden
        self.num_residual_layers = num_residual_layers
        self.dim_residual_hidden = dim_residual_hidden
    
        self._dec1 = nn.Conv2d(
            in_channels, 
            dim_hidden,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self._residual_stack = ResidualStack(
            dim_hidden, 
            dim_hidden,
            num_residual_layers,
            dim_residual_hidden
        )
        self._dec2 = nn.ConvTranspose2d(
            dim_hidden, 
            dim_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self._dec3 = nn.ConvTranspose2d(
            dim_hidden // 2, 
            3,
            kernel_size=4,
            stride=2,
            padding=1
        )
 
    def forward(self, z_q):
        h = self._dec1(z_q)
        h = self._residual_stack(h)
        h = torch.relu(self._dec2(h))
        x_reconstructed = self._dec3(h)
        x_reconstructed = torch.sigmoid(x_reconstructed)
        return x_reconstructed


class VQVAE(nn.Module):

    def __init__(self, encoder, decoder, vector_quantizer, pre_vq_conv, data_variance, name=None):
        super(VQVAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vector_quantizer = vector_quantizer
        self._pre_vq_conv = pre_vq_conv
        self._data_variance = data_variance
        
    def forward(self, inputs):
        z = self._pre_vq_conv(self._encoder(inputs))
        vq_output = self._vector_quantizer(z)
        x_reconstructed = self._decoder(vq_output['quantize'])
        reconstructed_error = torch.mean(
            torch.square(x_reconstructed - inputs) / torch.tensor(self._data_variance))
        loss = reconstructed_error + vq_output['vq_loss']
        return {
            'z': z,
            'x_reconstructed': x_reconstructed,
            'loss': loss,
            'reconstructed_error': reconstructed_error,
            'vq_output': vq_output,
        }