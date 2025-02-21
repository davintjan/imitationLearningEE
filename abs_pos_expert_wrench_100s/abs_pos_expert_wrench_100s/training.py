import argparse
import copy
import os
# import pytorch3d.transforms
import sys
import torch
from datetime import datetime
from enum import Enum
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from gen_dataset import *
from Model import *
import random
from torch.cuda import is_available as cuda_available
from matplotlib import pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on the RelativePoseDataset or make predictions")
    parser.add_argument('--model_name', type=str, default='FFBC', help='Name of Model')
    parser.add_argument('--train_relative_directory', type=str, default='../../expert_ML/training_dataset', help='Directory containing data files')
    parser.add_argument('--test_relative_directory', type=str, default='../../expert_ML/test_dataset',
                        help='Directory containing data files')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=70, help='Number of epochs to train')
    parser.add_argument('--model_path', type=str, default='runs/FFBC_model.pth', help='Path to the saved model for prediction')
    return parser.parse_args()


def train_and_evaluate(dataloader, model, criterion, optimizer, num_epochs, writer, test_dataloader):
    train_iteration = 0
    eval_iteration = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch"):
            # Ensure inputs and labels are of type float32
            inputs, labels = inputs.to(torch.float32).to(next(model.parameters()).device), labels.to(torch.float32).to(next(model.parameters()).device)
            # print(f"Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Outputs shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            writer.add_scalar('Training Loss', loss, train_iteration)
            train_iteration += 1

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.10f}")

        # Evaluate the model
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Evaluation", unit="batch"):
                # Ensure inputs and labels are of type float32
                inputs, labels = inputs.to(torch.float32).to(next(model.parameters()).device), labels.to(torch.float32).to(next(model.parameters()).device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.10f}")
        writer.add_scalar('Test Loss', avg_test_loss, epoch, eval_iteration)
        eval_iteration += 1

        # Save the model
        model_save_path = f"runs_FFBC_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")



def predict(input_data, model):
    model.eval()
    inputs = torch.tensor(input_data).float()
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.numpy()

def main():
    args = parse_args()

    # Set up TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/{args.model_name}_{timestamp}"
    writer = SummaryWriter(log_dir)

    # Create training and testing datasets
    train_dataset = RelativePoseDataset(relative_directory=args.train_relative_directory)
    test_dataset = RelativePoseDataset(relative_directory=args.test_relative_directory)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FFBC()
    # Check if CUDA is available and use it
    device = torch.device("cuda" if cuda_available() else "cpu")
    model.to(device)
    criterion = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training and evaluation...")
    train_and_evaluate(train_dataloader, model, criterion, optimizer, args.num_epochs, writer, test_dataloader)

    # Close the TensorBoard writer to flush remaining events to disk
    writer.close()
if __name__ == "__main__":
    main()