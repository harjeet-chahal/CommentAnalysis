
import numpy as np
import random
import copy

def get_length_buckets(texts):
    """
    Returns boolean indices for Short (<20 words) and Long (>100 words) comments.
    """
    lengths = np.array([len(t.split()) for t in texts])
    short_indices = np.where(lengths < 20)[0]
    long_indices = np.where(lengths > 100)[0]
    
    return short_indices, long_indices

def inject_noise_string(text, noise_level=0.1):
    """
    Swaps 10% (noise_level) of characters in the string randomly.
    """
    if not text or len(text) < 2:
        return text
    
    chars = list(text)
    num_swaps = int(len(chars) * noise_level)
    
    for _ in range(num_swaps):
        idx1 = random.randint(0, len(chars) - 1)
        idx2 = random.randint(0, len(chars) - 1)
        # Swap
        chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
        
    return "".join(chars)

def get_noisy_dataset(texts, noise_level=0.1):
    """
    Applies noise to a list of texts.
    """
    noisy_texts = [inject_noise_string(t, noise_level) for t in texts]
    return noisy_texts
