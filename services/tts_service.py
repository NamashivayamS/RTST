import numpy as np
from models.indic_f5_model import generate_tamil_speech

class TTSService:
    """
    Standardized wrapper service for the IndicF5 Text-To-Speech engine.
    This service is designed to be called asynchronously by the background worker thread.
    """
    def __init__(self):
        # Model is already pre-loaded into memory by indic_f5_model import
        print("TTSService initialized and ready.")

    def generate_audio(self, text: str) -> tuple[np.ndarray, int]:
        """
        Generates audio for a given text chunk.
        Returns:
            audio_data (np.ndarray): The raw PCM audio array.
            sample_rate (int): The sample rate (24000 for IndicF5).
        """
        if not text.strip():
            # Return empty audio for empty chunks to prevent model crash
            return np.array([]), 24000
            
        try:
            # Call our heavily optimized and tuned F5 generation function
            audio_arr, sr = generate_tamil_speech(text)
            return audio_arr, sr
        except Exception as e:
            print(f"[TTSService Error] Failed to generate audio for chunk '{text}': {e}")
            # Fallback to silence on failure to keep pipeline alive
            return np.zeros(24000, dtype=np.float32), 24000

# Quick test when run directly
if __name__ == "__main__":
    service = TTSService()
    arr, sr = service.generate_audio("சோதனை.")
    print(f"Generated {len(arr)} samples at {sr}Hz.")
