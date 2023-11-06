import os

learning_rates = ['0.00001', '0.0001', '0.001']
batch_sizes = ['16', '32', '64', '128', '256']
model_names = ['LSTM_CNN', 'GRU2', 'GRU2_CNN', 'LSTM']
metrics = ['Test loss', 'Test accuracy', 'L1 Score', 'Precision', 'Recall', 'Average Precision']

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# Create a dictionary to store the results
results = {}

# Loop through the folders and extract the information
for lr in learning_rates:
    results[lr] = {}
    for batch_size in batch_sizes:
        results[lr][batch_size] = {}
        for model_name in model_names:
            folder_path = script_dir+ '\\'+ lr + '\\'+ batch_size + '\\'+ model_name
            file_path = folder_path+ '\\average_precision.txt'



            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = [float(line.strip().split(': ')[1]) for line in lines[1:7]]

            results[lr][batch_size][model_name] = values

# Print the results in a table format
print(f"{'Model':<12}{'Batch Size':<8}{'Learning Rate':<10}", end="")
for metric in metrics:
    print(f"{metric:<20}", end="")
print()

for lr in learning_rates:
    for batch_size in batch_sizes:
        for model_name in model_names:
            print(f"{model_name:<12}{batch_size:<8}{lr:<10}", end="")
            for value in results[lr][batch_size][model_name]:
                print(f"{value:<20.4f}", end="")
            print()
