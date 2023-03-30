import config
import whisper
import os, glob
import sounddevice as sd
import wavio as wv
import datetime
import openai

openai.api_key = open("key.txt", "r").read()
openai.proxy = 'http://127.0.0.1:7890'

message_history = [{"role": "assistant", "content": f"OK"}]
def GPT(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"帮我用中文总结以下的对话: {input}"}) # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content




freq = 44100 # frequency change
duration = 5 # in seconds
print('----------Recording----------')
# find most recent files in a directory
recordings_dir = os.path.join('recordings', '*')

model = whisper.load_model("base")

# list to store which wav files have been transcribed
transcribed = []


while True:
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y_%m_%d_%H_%M_%S")
    print(filename)

    # Start recorder with the given values of duration and sample frequency
    # PTL Note: I had to change the channels value in the original code to fix a bug
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()

    # Convert the NumPy array to audio file
    wv.write(f"./recordings/{filename}.wav", recording, freq, sampwidth=2)
    # get most recent wav recording in the recordings directory
    files = sorted(glob.iglob(recordings_dir), key=os.path.getctime, reverse=True)
    if len(files) < 1:
        continue

    latest_recording = files[0]
    latest_recording_filename = latest_recording.split('_')[1]

    if os.path.exists(latest_recording) and not latest_recording in transcribed:
        audio = whisper.load_audio(latest_recording)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)

        result = whisper.decode(model, mel, options)

        if result.no_speech_prob < 0.5:
            print(result.text)
            # append text to transcript file
            with open(config.TRANSCRIPT_FILE, 'a') as f:
                f.write(result.text)

            # save list of transcribed recordings so that we don't transcribe the same one again
            transcribed.append(latest_recording)
        #triggering phrase for GPT language model
        if '幫我總結' in result.text:
            print("--------Deploying Jarvis--------")
            transcript = open('./transcriptions/transcript.txt', 'r').read()
            print(GPT(transcript))



