


from tensorboard.backend.event_processing import event_accumulator


train_log_dir = "logs/fit/20241123-021809/train"
validation_log_dir = "logs/fit/20241123-021809/validation"

def extract_tensors(log_dir, tag_name):
    # Load the TensorBoard logs
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    try:
        # Extract tensor data for the given tag
        tensors = ea.Tensors(tag_name)
        print(f"Data for tag '{tag_name}':")
        for tensor in tensors:
            print(f"Step: {tensor.step}, Value: {tensor.tensor_proto}")  # Prints raw tensor
    except KeyError:
        print(f"Tag '{tag_name}' not found in {log_dir}")

# List of tensor tags to extract (you can modify this list)
train_tensor_tags = ['epoch_f1_score', 'epoch_loss', 'epoch_recall', 'epoch_learning_rate']
validation_tensor_tags = ['evaluation_f1_score_vs_iterations', 'evaluation_loss_vs_iterations', 'evaluation_recall_vs_iterations']

# Extract tensors from train logs
print("TRAIN LOG TENSORS:")
for tag in train_tensor_tags:
    extract_tensors(train_log_dir, tag)

# Extract tensors from validation logs
print("\nVALIDATION LOG TENSORS:")
for tag in validation_tensor_tags:
    extract_tensors(validation_log_dir, tag)









import tensorflow as tf
from tensorboard.util import tensor_util

def extract_tensors_readable(log_dir, tag_name):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    try:
        tensors = ea.Tensors(tag_name)
        print(f"Data for tag '{tag_name}':")
        for tensor in tensors:
            step = tensor.step
            value = tensor_util.make_ndarray(tensor.tensor_proto)  # Converts raw tensor to NumPy array
            print(f"Step: {step}, Value: {value}")
    except KeyError:
        print(f"Tag '{tag_name}' not found in {log_dir}")

# Extract readable tensors
print("Readable TRAIN LOG TENSORS:")
for tag in train_tensor_tags:
    extract_tensors_readable(train_log_dir, tag)

print("\nReadable VALIDATION LOG TENSORS:")
for tag in validation_tensor_tags:
    extract_tensors_readable(validation_log_dir, tag)
