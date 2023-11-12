import torch
import torch.nn as nn

def train_and_evaluate_model(model, train_loader, test_loader, num_epochs=10):
    # Criterion for computing the loss during training
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer for updating model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train() # Training mode
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval() # Evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.2f}%')
