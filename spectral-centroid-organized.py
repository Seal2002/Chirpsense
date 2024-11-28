import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import soundfile as sf

def create_spectral_centroid(audio_path, npy_path, img_path, filename, fixed_length=30, sr=22050):
    """
    Create and save Spectral Centroid features with fixed dimensions for 30 seconds
    
    Parameters:
    - audio_path: path to audio file
    - npy_path: path to save numpy arrays
    - img_path: path to save visualization images
    - filename: name of the file
    - fixed_length: duration in seconds to process (default 30 seconds)
    - sr: sample rate (default 22050 Hz)
    """
    try:
        # Calculate fixed number of samples for the desired duration
        fixed_samples = int(fixed_length * sr)
        
        # Load and process audio
        y, _ = librosa.load(audio_path, sr=sr)
        
        # Handle audio length
        if len(y) < fixed_samples:
            # For shorter audio, center it in the 30-second window
            pad_length = fixed_samples - len(y)
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            # If audio is longer, take the middle 30 seconds
            start_idx = (len(y) - fixed_samples) // 2
            y = y[start_idx:start_idx + fixed_samples]
        
        # Calculate Spectral Centroid
        hop_length = 512
        n_fft = 2048
        
        # Calculate spectral centroid and its delta
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Calculate delta centroid
        centroid_delta = librosa.feature.delta(centroid)
        
        # Combine features
        combined_features = np.concatenate([centroid, centroid_delta], axis=0)
        
        # Normalize features
        normalized_features = (combined_features - np.mean(combined_features, axis=1, keepdims=True)) / \
                            (np.std(combined_features, axis=1, keepdims=True) + 1e-6)
        
        # Save the raw feature data
        os.makedirs(os.path.join(npy_path, os.path.dirname(filename)), exist_ok=True)
        save_filename_npy = os.path.join(npy_path, f"{filename}.npy")
        np.save(save_filename_npy, normalized_features)
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Create time array for x-axis
        times = librosa.times_like(centroid, sr=sr, hop_length=hop_length)
        
        # Plot Spectral Centroid
        plt.subplot(2, 1, 1)
        plt.semilogy(times, centroid[0], label='Spectral Centroid')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Hz')
        plt.title('Spectral Centroid')
        plt.legend()
        
        # Plot Delta Centroid
        plt.subplot(2, 1, 2)
        plt.plot(times, centroid_delta[0], label='Delta Centroid', color='orange')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Delta')
        plt.title('Spectral Centroid Delta')
        plt.legend()
        
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.join(img_path, os.path.dirname(filename)), exist_ok=True)
        save_filename_png = os.path.join(img_path, f"{filename}.png")
        plt.savefig(save_filename_png, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Clear memory
        del y, centroid, centroid_delta, normalized_features
        gc.collect()
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        plt.close('all')
        gc.collect()

def process_dataset(input_path, output_base_path, batch_size=5):
    """
    Process all audio files in the dataset with batched processing
    """
    # Create separate directories for numpy arrays and images
    npy_path = os.path.join(output_base_path, 'numpy_features')
    img_path = os.path.join(output_base_path, 'visualizations')
    
    os.makedirs(npy_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    
    print(f"Numpy arrays will be saved to: {npy_path}")
    print(f"Visualizations will be saved to: {img_path}")
    
    # Get all MP3 files recursively
    audio_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Calculate and display expected dimensions
    sr = 22050
    fixed_length = 30
    hop_length = 512
    n_frames = int(np.ceil(fixed_length * sr / hop_length))
    n_features = 2  # Centroid and delta
    print(f"\nExpected Spectral Centroid dimensions for each file:")
    print(f"Number of features: {n_features}")
    print(f"Time steps: {n_frames}")
    print(f"Final shape: ({n_features}, {n_frames})")
    
    # Track audio lengths for analysis
    audio_lengths = []
    
    # Process files in batches
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(audio_files)//batch_size + 1}")
        
        for audio_file in tqdm(batch):
            # Create output filename (preserve folder structure)
            rel_path = os.path.relpath(audio_file, input_path)
            filename = os.path.splitext(rel_path)[0]
            
            # Get audio length before processing
            try:
                with sf.SoundFile(audio_file) as f:
                    duration = len(f) / f.samplerate
                    audio_lengths.append(duration)
            except Exception as e:
                print(f"Could not get duration for {audio_file}: {str(e)}")
            
            create_spectral_centroid(audio_file, npy_path, img_path, filename)
            
            # Force garbage collection after each file
            plt.close('all')
            gc.collect()
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Print audio length statistics
    audio_lengths = np.array(audio_lengths)
    print("\nAudio Length Statistics:")
    print(f"Mean duration: {np.mean(audio_lengths):.2f} seconds")
    print(f"Median duration: {np.median(audio_lengths):.2f} seconds")
    print(f"Min duration: {np.min(audio_lengths):.2f} seconds")
    print(f"Max duration: {np.max(audio_lengths):.2f} seconds")
    print(f"Number of files shorter than 30s: {np.sum(audio_lengths < 30)}")
    print(f"Number of files longer than 30s: {np.sum(audio_lengths > 30)}")

if __name__ == "__main__":
    # Disable librosa cache
    os.environ['LIBROSA_CACHE_DIR'] = ''
    
    # Configure matplotlib to use Agg backend
    plt.switch_backend('Agg')
    
    input_path = r"D:\Dataset\New\train_audio"
    output_base_path = r"D:\Capstone\New\Spectral Centroid"
    
    try:
        process_dataset(input_path, output_base_path, batch_size=5)
    except MemoryError:
        print("\nMemory error occurred. Retrying with single file processing...")
        process_dataset(input_path, output_base_path, batch_size=1)
