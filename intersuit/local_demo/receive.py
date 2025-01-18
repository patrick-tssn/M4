import cv2
import pyaudio
import socket
import struct
import pickle
import numpy as np

# Configure socket
host_ip = '0.0.0.0'  # Listen on all available interfaces
host_port = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host_ip, host_port))
sock.listen(5)
print(f"Listening on {host_ip}:{host_port}")

conn, addr = sock.accept()
print(f"Connection from {addr}")

# Audio Playback (PyAudio)
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)

data_buffer = b""  # Buffer to hold incoming data

try:
    while True:
        # Receive video frame size
        while len(data_buffer) < struct.calcsize("Q"):
            packet = conn.recv(4096)  # Receive data in chunks
            if not packet:
                break
            data_buffer += packet
        if len(data_buffer) < struct.calcsize("Q"):
            break  # Connection closed
        frame_size = struct.unpack("Q", data_buffer[:struct.calcsize("Q")])[0]
        data_buffer = data_buffer[struct.calcsize("Q"):]

        # Receive the actual video frame
        while len(data_buffer) < frame_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data_buffer += packet
        if len(data_buffer) < frame_size:
            break  # Connection closed
        frame_data = data_buffer[:frame_size]
        data_buffer = data_buffer[frame_size:]

        # Decode video frame
        try:
            frame = pickle.loads(frame_data)
        except pickle.UnpicklingError as e:
            print(f"Failed to unpickle frame data: {e}")
            continue

        # Convert the frame to a NumPy array and display it
        frame = np.array(frame)
        # cv2.imshow("Video", frame)
        print("frame: ", frame.shape)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Receive audio data size
        while len(data_buffer) < struct.calcsize("Q"):
            packet = conn.recv(4096)
            if not packet:
                break
            data_buffer += packet
        if len(data_buffer) < struct.calcsize("Q"):
            break  # Connection closed
        audio_size = struct.unpack("Q", data_buffer[:struct.calcsize("Q")])[0]
        data_buffer = data_buffer[struct.calcsize("Q"):]

        # Receive the actual audio data
        while len(data_buffer) < audio_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data_buffer += packet
        if len(data_buffer) < audio_size:
            break  # Connection closed
        audio_data = data_buffer[:audio_size]
        data_buffer = data_buffer[audio_size:]

        # # Play audio
        # stream.write(audio_data)

except KeyboardInterrupt:
    print("Streaming stopped.")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    conn.close()
    sock.close()
    cv2.destroyAllWindows()
