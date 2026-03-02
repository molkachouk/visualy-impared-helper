import subprocess
import os
import tempfile
import pygame
import threading
import queue
import time
import re

class AudioOutput:
    def __init__(self, speed=1.0, max_queue=5):
        pygame.mixer.init()
        self.speed = speed

        # 1. SET MANUAL PATH TO DOWNLOADED PIPER 
        # Update this to the exact folder where you extracted the zip
        self.piper_exe = r"C:\Users\asus\Documents\BlindAssistance\piper\piper.exe"
        
        # 2. SET MODEL PATH 
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "voices", "en_US-lessac-medium.onnx")

        # Validation
        if not os.path.exists(self.piper_exe):
            print(f"CRITICAL ERROR: piper.exe still not found at {self.piper_exe}")
        if not os.path.exists(self.model_path):
            print(f"CRITICAL ERROR: Model not found at {self.model_path}")

        self.last_text = ""
        self.audio_queue = queue.Queue(maxsize=max_queue)
        threading.Thread(target=self._process_queue, daemon=True).start()

    def _normalize_text(self, text):
        return re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text).strip()

    def speak(self, text, priority=False):
        if not text: return
        text = self._normalize_text(text)
        if text == self.last_text: return
        self.last_text = text

        if priority:
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            pygame.mixer.music.stop()

        if not self.audio_queue.full():
            self.audio_queue.put(text)

    def _process_queue(self):
        while True:
            text = self.audio_queue.get()
            temp_path = None
            try:
                # Create a temp file
                fd, temp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd) # Close handle so Piper can write to it

                # Run Piper and tell it to output to the temp file
                cmd = [
                    self.piper_exe,
                    "--model", self.model_path,
                    "--output_file", temp_path,
                    "--length_scale", str(self.speed)
                ]

                # Use shell=True on Windows to avoid some subprocess issues
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                process.communicate(input=text.encode('utf-8'))

                # Play via pygame-ce
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                
                # Unload music so we can delete the file
                pygame.mixer.music.unload()

            except Exception as e:
                print(f"Audio Error: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
            
            self.audio_queue.task_done()