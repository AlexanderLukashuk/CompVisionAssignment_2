import torch
import dataloader
import models
import train_eval

if __name__ == '__main__':
    # Load dataset
    train_loader = dataloader.load_and_preprocess_data("Agricultural-crops", batch_size=64, shuffle=True)

    # Number of classes in dataset
    num_classes = len(train_loader.dataset.classes)

    # Training models with different variations
    # Model with Sigmoid activation
    model_sigmoid = models.CNN(activation_fn=torch.nn.Sigmoid(), num_classes=num_classes)

    # Load the test result for evaluation
    test_loader = dataloader.load_and_preprocess_data("Agricultural-crops", batch_size=64, shuffle=False)

    # Training and evaluating the model with Sigmoid activation
    train_eval.train_and_evaluate_model(model_sigmoid, train_loader, test_loader)

    # Model with ReLU activation
    model_relu = models.CNN(activation_fn=torch.nn.ReLU(), num_classes=num_classes)

    # Training and evaluating the model with ReLU activation
    train_eval.train_and_evaluate_model(model_relu, train_loader, test_loader)
