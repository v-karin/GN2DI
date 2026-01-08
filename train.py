import time
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import trackio as wandb

from config import get_args
from data.dataset import get_dataset, DataSet
from model.gn2di import GN2DI
from utils.graph import construct_graph

def train(args):
    # Set up wandb
    wandb.init(project="GN2DI", config=args)
    
    # Get data
    data = get_dataset(args.dataset)
    
    # Split data
    spline1 = int(data.shape[0] * 0.7)
    spline2 = int(data.shape[0] * 0.8)
    
    train_data = data[:spline1]
    valid_data = data[spline1:spline2]
    test_data = data[spline2:]
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate static features
    rng_static = np.random.RandomState(seed=args.seed)
    mean_var = [np.mean(train_data, axis=0).tolist(), np.std(train_data, axis=0).tolist()]
    tensor_mean = torch.tensor(np.mean(train_data, axis=0), dtype=torch.float32).to(device)
    
    sample_normal = [
        [rng_static.normal(m, s, size=args.in_channel_gl).tolist()]
        for m, s in zip(mean_var[0], mean_var[1])
    ]
    x_static = torch.FloatTensor(sample_normal).reshape(-1, args.in_channel_gl).to(device)
    
    # Initialize model
    model = GN2DI(
        args.num_initial_weights, 
        args.hidden_dim_pre_weight, 
        args.num_lay_pre_weight, 
        args.in_channel_gl,
        args.num_conv_lay_gl,
        args.hidden_dim_conv_gl,
        args.hidden_dim_gl, 
        args.in_channel_imp,
        args.num_conv_lay_imp,
        args.hidden_dim_conv_imp,
        args.hidden_dim_readout_imp,
        args.hidden_dim_updater,
        args.dropout_pre_weight,
        args.dropout_gl,
        args.dropout_imp
    ).to(device)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(seed=args.seed)
    ot = time.time()    

    # Training loop
    for epoch in tqdm(range(args.max_epoch), desc="Training..."):
        # Construct graph
        edge_index, edge_weight = construct_graph(train_data, args.k_neigh)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device) 
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32).to(device) 
        
        # Generate masks
        mask_train = torch.tensor(
            rng.binomial(1, 1 - 0.3, size=(train_data.shape[0], train_data.shape[1]))
        ).type(torch.bool).to(device)
        
        mask_val = torch.tensor(
            rng.binomial(1, 1 - 0.3, size=(valid_data.shape[0], valid_data.shape[1]))
        ).type(torch.bool).to(device)
        
        # Convert data to tensors
        train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
        val_tensor = torch.tensor(valid_data, dtype=torch.float32).to(device)
        
        # Create datasets and dataloaders
        dataset_train = DataSet(train_tensor, mask_train)
        dataset_val = DataSet(val_tensor, mask_val)
        
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size)
        val_loader = DataLoader(dataset_val, batch_size=args.batch_size)
        
        # Training step
        model.train()
        train_losses = []
        for data, mask in train_loader:
            optimizer.zero_grad()
            
            x_dynamic = torch.where(
                mask,
                tensor_mean.repeat(data.shape[0], 1),
                data
            )
            x_dynamic = x_dynamic.unsqueeze(dim=1).repeat(1, 1, args.in_channel_imp)

            print(x_static.device, x_dynamic.device, edge_index.device, edge_weight.device)
            x_rec = model(x_static, x_dynamic, edge_index, edge_weight)
            loss = criterion(x_dynamic[mask == False], x_rec[mask == False])
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for data, mask in val_loader:
                x_dynamic = torch.where(
                    mask,
                    tensor_mean.repeat(data.shape[0], 1),
                    data
                )
                x_dynamic = x_dynamic.unsqueeze(dim=1).repeat(1, 1, args.in_channel_imp)
                x_rec = model(x_static, x_dynamic, edge_index, edge_weight)
                
                loss = criterion(x_dynamic[mask == False], x_rec[mask == False])
                val_losses.append(loss.item())
        
        # Log metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        nt = time.time() - ot

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": nt,
        })
        
        print(f"Epoch {epoch}: Train MSE = {train_loss:.4f}, Val MSE = {val_loss:.4f}, Time Elapsed = {nt:.6f}")

if __name__ == "__main__":
    args = get_args()
    train(args)
