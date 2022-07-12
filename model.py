import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):

    def __init__(self, in_channels, dim_hidden, dim_residual_hidden):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=dim_residual_hidden,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=dim_residual_hidden,
                      out_channels=dim_hidden,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):

    def __init__(self, in_channels, dim_hidden, num_residual_layers, dim_residual_hidden):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, dim_hidden, dim_residual_hidden)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


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
            dim_hidden // 2, 
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

    def __init__(self, dim_embedding, num_embeddings, commitment_cost=None):
        super(VectorQuantizer, self).__init__()
        self._dim_embedding = dim_embedding
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(num_embeddings, dim_embedding)
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self._commitment_cost = commitment_cost
     
    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._dim_embedding)
        
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
        
        # For Loss
        quantize_for_loss = quantized
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # vq_loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return {
            'distances': distances,
            'encodings': encodings,
            'encoding_indices': encoding_indices,
            'quantized': quantized,
            'quantize_for_loss':quantize_for_loss,
            'perplexity': perplexity,
            # 'vq_loss': vq_loss, 
        }


class Decoder(nn.Module):
    def __init__(self, in_channels, dim_hidden, num_residual_layers, dim_residual_hidden):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=dim_hidden,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=dim_hidden,
                                             dim_hidden=dim_hidden,
                                             num_residual_layers=num_residual_layers,
                                             dim_residual_hidden=dim_residual_hidden)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=dim_hidden, 
                                                out_channels=dim_hidden//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=dim_hidden//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)


class VQVAE(nn.Module):

    def __init__(self, encoder, decoder, vector_quantizer, pre_vq_conv, data_variance=None, name=None):
        super(VQVAE, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vector_quantizer = vector_quantizer
        self._pre_vq_conv = pre_vq_conv
        self._data_variance = data_variance
        
    def forward(self, inputs):
        z = self._pre_vq_conv(self._encoder(inputs))
        vq_output = self._vector_quantizer(z)
        z_q = vq_output['quantized']
        x_reconstructed = self._decoder(z_q)
        # reconstructed_error = (
        #     F.mse_loss(x_reconstructed, inputs) / torch.tensor(self._data_variance))
        # loss = reconstructed_error + vq_output['vq_loss']
        return {
            'z': z,
            'x_reconstructed': x_reconstructed,
            # 'loss': loss,
            # 'reconstructed_error': reconstructed_error,
            'vq_output': vq_output,
        }