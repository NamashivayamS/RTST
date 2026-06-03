import time
import sys

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Loading core PyTorch libraries first to prevent Windows DLL conflicts...")
import torch
import soundfile as sf

print("Loading services...")
from services.router_service import RouterService

print("Initializing Router...")
router = RouterService()

# Simulate a translation arriving from the STT -> Translation pipeline
test_translation = "காலை வணக்கம் அனைவருக்கும். இன்று நாம் நிகழ்நேர பேச்சு மொழிபெயர்ப்பு அமைப்பை சோதிக்கிறோம்."

# Process it instantly (returns almost instantly)
start_t = time.time()
print("\n--- Sending Translation to Router ---")
router.process_translation(test_translation)
print(f"Router returned in {time.time() - start_t:.4f} seconds! (Frontend is unblocked)\n")

# Now simulate the playback system grabbing the audio sequentially
print("Simulating Audio Playback Handler...")
for i in range(3): # We expect 3 chunks based on chunking logic
    print(f"Waiting for audio chunk {i+1} from background thread...")
    
    # Wait for up to 60 seconds for the TTS engine to finish generating the chunk
    audio_payload = router.get_generated_audio(block=True, timeout=60)
    
    if audio_payload is None:
        print("TIMEOUT: Did not receive audio chunk!")
        break
        
    output_file = f"output_stream_chunk_{i+1}.wav"
    sf.write(output_file, audio_payload["audio"], audio_payload["sample_rate"])
    
    print(f"✅ Received Audio Chunk {i+1} for text: '{audio_payload['text']}'")
    print(f"   Saved to {output_file}")
    
print("\nSimulation Complete!")
router.shutdown()
