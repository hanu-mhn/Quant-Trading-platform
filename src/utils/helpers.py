def log_message(message):
    """Logs a message to the console."""
    print(f"[LOG] {message}")

def calculate_moving_average(data, window):
    """Calculates the moving average of a given data set."""
    if len(data) < window:
        return None
    return sum(data[-window:]) / window

def normalize_data(data):
    """Normalizes the data to a range of 0 to 1."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def split_data(data, train_size):
    """Splits the data into training and testing sets."""
    train_length = int(len(data) * train_size)
    return data[:train_length], data[train_length:]