import time
import threading
from queue import Queue, Empty
import torch

from utils import get_completion
from main import model


class ChatInterface:
    def __init__(self, model_func, user_prefix="User", model_prefix="Model"):
        self.model_func = model_func
        self.user_prefix = user_prefix
        self.model_prefix = model_prefix
        self.setup_complete = False
        self.output_queue = Queue()
        self.stop_event = threading.Event()

    def _stream_output(self):
        while not self.stop_event.is_set():
            try:
                word = self.output_queue.get_nowait()
                print(word, end='', flush=True)
            except Empty:
                time.sleep(0.05)

    def _run_model(self, prompt):
        for word in self.model_func(prompt):
            if self.stop_event.is_set():
                break
            self.output_queue.put(word)

        self.output_queue.put("\n")

    def chat_loop(self):
        if not self.setup_complete:
            self.setup_complete = True
            print("Ready! Type your message or ':h' for commands")

        while True:
            try:
                user_input = input(f"\n{self.user_prefix}> ").strip()

                if user_input.startswith(':'):
                    cmd = user_input[1:].lower()
                    if cmd == "q":
                        print("Ending chat session.")
                        break
                    elif cmd == 'h':
                        print("\nAvailable commands:")
                        print(" :h - Show commands")
                        print(" :q - End the chat")
                        print(" :cls - Clear screen\n")
                    elif cmd == 'cls':
                        print("\033c", end="")
                    continue

                print(f"{self.model_prefix}> ", end='', flush=True)
                self.stop_event.clear()
                stream_thread = threading.Thread(target=self._stream_output)
                stream_thread.start()

                self._run_model(user_input)

                self.stop_event.set()
                stream_thread.join()

            except KeyboardInterrupt:
                print("\nUse :q to exit or :h for commands")
                self.stop_event.set()
            except EOFError:
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    print("Initializing session...")
    def get_completion_stream(prompt):
        device = "mps" if torch.mps.is_available() else "cpu"
        return get_completion(model, prompt, device=device, temperature=0.1)

    chat = ChatInterface(
        model_func=get_completion_stream,
        user_prefix="You",
        model_prefix="Model"
    )
    chat.chat_loop()