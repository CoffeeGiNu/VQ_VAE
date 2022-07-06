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
parser.add_argument('-nes', '--num_embeddings', default=256, type=int)
parser.add_argument('-cc', '--commitment_cost', default=0.25, type=float)
parser.add_argument('-d', '--decay', default=0.99, type=float)
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)

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


if __name__ == "__main__":
    fix_seed(SEED)
    torch.cuda.empty_cache()
    log_dir = "./logs"
    writer = SummaryWriter(log_dir)
    loss_fn = lambda lower_bound: -sum(lower_bound)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(log_dir) if not os.path.exists(log_dir) else None
    
    dataset_train = load_dataset_folder(path='datasets/', split='train')
    dataset_valid = load_dataset_folder(path='datasets/', split='valid', shuffle=False)
    for sample_train in dataset_train.as_numpy_iterator():
        sample_inputs_train = sample_train['image']
        break
    train_data_variance = torch.var(torch.tensor(sample_inputs_train)).numpy()
    
    model = VQVAE(
        encoder=Encoder(
            in_channels=IN_CHANNELS,
            dim_hidden=DIM_HIDDEN,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            dim_residual_hidden=DIM_RESIDUAL_HIDDEN,
        ),
        decoder=Decoder(
            in_channels=DIM_EMBEDDING,
            dim_hidden=DIM_HIDDEN,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            dim_residual_hidden=DIM_RESIDUAL_HIDDEN,
        ),
        vector_quantizer=VectorQuantizer(
            dim_embedding=DIM_EMBEDDING, 
            num_embeddings=NUM_EMBEDDINGS, 
            # commitment_cost=COMMITMENT_COST,
        ),
        pre_vq_conv=nn.Conv2d(
            DIM_HIDDEN, 
            DIM_EMBEDDING,
            kernel_size=1,
            stride=1
        ),
        # data_variance=train_data_variance,
    ).to(device)
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr=LEARNING_RATE, 
    #     # weight_decay=DECAY, 
    #     eps=0.001
    # )
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        # weight_decay=DECAY, 
        # eps=0.001
    )
    earlystopping = EarlyStopping(path='models/', patience=5)
    criterion = VQVAELoss(COMMITMENT_COST, train_data_variance)

    with torch.profiler.profile(
        # schedule=torch.profiler.schedule(
        #     wait=2,
        #     warmup=2,
        #     active=6,
        #     repeat=1),
        # use_cuda=(True if device=='cuda' else False),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        with_stack=True
    ) as profiler:
        for e in range(NUM_EPOCHS):
            model = epoch_loop(model, dataset_train, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=True, profiler=profiler, writer=writer)
            model = epoch_loop(model, dataset_valid, optimizer, criterion, device, e, NUM_EPOCHS, BATCH_SIZE, is_train=False, earlystopping=earlystopping, profiler=profiler, writer=writer)

            if earlystopping.early_stop:
                # writer.add_graph(model)
                writer.close()
                break
        writer.close()