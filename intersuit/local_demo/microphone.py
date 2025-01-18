# import ffmpeg
# import numpy as np
# import pyaudio
# import wave
# import time

# # Define audio stream parameters
# FORMAT = pyaudio.paInt16
# CHANNELS = 2
# RATE = 44100
# CHUNK = 4096
# DURATION = 5  # seconds
# OUTPUT_FILENAME = 'output.wav'

# def record_audio():
#     # Initialize PyAudio
#     p = pyaudio.PyAudio()

#     # Create a stream to play audio
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     output=True)

#     # Use FFmpeg to capture the audio stream
#     process = (
#         ffmpeg
#         .input('http://10.1.101.4:1234', format='mp3', timeout='2000000')  # Increased timeout
#         .output('pipe:', format='s16le', acodec='pcm_s16le', ac=CHANNELS, ar=str(RATE))
#         .global_args('-re')
#         .run_async(pipe_stdout=True, pipe_stderr=True)
#     )

#     frames = []

#     try:
#         start_time = time.time()
#         while time.time() - start_time < DURATION:
#             in_data = process.stdout.read(CHUNK)
#             if not in_data:
#                 break
#             stream.write(in_data)
#             frames.append(in_data)
#     except ffmpeg.Error as e:
#         print(f"FFmpeg error: {e}")
#         print(process.stderr.read().decode())  # Display FFmpeg error messages
#     except KeyboardInterrupt:
#         pass
#     finally:
#         stream.stop_stream()
#         stream.close()
#         p.terminate()
#         process.terminate()

#     # Save the captured audio to a WAV file
#     with wave.open(OUTPUT_FILENAME, 'wb') as wf:
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))

#     print(f"Audio saved to {OUTPUT_FILENAME}")

# if __name__ == '__main__':
#     record_audio()


import socket
import wave
import numpy as np

# Configuration
# SERVER_IP = "10.1.101.4"  # Listen on all interfaces
SERVER_IP = "0.0.0.0"  # Listen on all interfaces
SERVER_PORT = 1234      # Port VLC is streaming to

# Create a socket to listen for the stream
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Use SOCK_DGRAM for RTP/UDP
sock.bind((SERVER_IP, SERVER_PORT))

print(f"Listening for audio stream on {SERVER_IP}:{SERVER_PORT}...")

# Open a WAV file to save the incoming audio
with wave.open("output.wav", "wb") as wav_file:
    # Set WAV parameters (modify based on your stream's format)
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16-bit audio
    wav_file.setframerate(44100)  # Sample rate

    while True:
        try:
            # Receive audio data
            data, addr = sock.recvfrom(4096)  # Buffer size of 4096
            print(f"Received {len(data)} bytes from {addr}")

            # Write raw audio data to WAV file
            wav_file.writeframes(data)

            # Process audio data (e.g., with NumPy)
            audio_array = np.frombuffer(data, dtype=np.int16)
            print(f"Processed Audio Data: {audio_array[:10]}")  # Example processing
        except KeyboardInterrupt:
            print("Stopping...")
            break
