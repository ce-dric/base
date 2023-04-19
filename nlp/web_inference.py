import gradio as gr
import random
import time
import numpy as np

from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# 하이퍼파라미터
seq_length = 5

with open('corpus.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))

char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

voca_size = len(chars)

# 모델 생성
model = Sequential()
model.add(SimpleRNN(64, input_shape=(seq_length, 1), activation="relu"))
model.add(Dense(voca_size, activation="softmax"))

model.load_weights('model/rnn.h5')

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = inference(history[-1][0])
        history[-1][1] = bot_message
        time.sleep(1)
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

def inference(message):
    seed = message
    generated_text = seed

    input_seq = np.array( [char_to_idx[char] for char in seed] )
    input_seq = input_seq.reshape(1, seq_length)


    for _ in range(50):
        predicted_idx = np.argmax( model.predict(input_seq, verbose=0) )
        predicted_char = idx_to_char[predicted_idx]
        generated_text += predicted_char
        
        input_seq = np.concatenate((input_seq[:, 1:], np.array(predicted_idx).reshape(1, 1)), axis=1)

    return generated_text
    

if __name__ == "__main__":
    demo.launch()
