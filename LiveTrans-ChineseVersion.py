import config
import whisper
import os, glob
import sounddevice as sd
import wavio as wv
import datetime
import openai

openai.api_key = open("key.txt", "r").read()
openai.proxy = 'http://127.0.0.1:7890'

'''
GPT语言模型的调用方式
'''
message_history = [{"role": "assistant", "content": f"OK"}]
def GPT(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"帮我用中文总结以下的对话: {input}"}) # 这段中文可以改成任意你希望达成的效果
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # 可以帮我们更好的联系上下文
      model="gpt-3.5-turbo", #便宜又够用的模型相比GPT4便宜10倍哦
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content




freq = 44100 # 音频的频率设置frequency
duration = 5 # 秒为单位
print('----------Recording----------')
# 找到最近创建的文档
recordings_dir = os.path.join('recordings', '*')

model = whisper.load_model("base") # 这边是调用本地的whisper模型，最快的是tiny，其次有base, medium, large等模型速度不一样精确程度也不一样

# 创建一个list，如果已经被transcribe过了就放进来给之后做过滤
transcribed = []

'''
1. 用sound device包将我们想录取的音频录制下来，并间隔5秒钟生成一个音频文件
2. 利用本地的whisper模型可以比api式的更快读取文件并进行语音转文字的操作
3. 每一个音频生成一段小文字，小文字全部都保存到transcript.txt文件中
4. 当系统发现小文字中有触发语句的时候，召唤GPT模型，并且把transcript.txt文件中的内容传送进入GPT模型
'''
while True:
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y_%m_%d_%H_%M_%S")
    print(filename)

    # 开始录音
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)

    # 触发以下句式能保证我们的录音文件只在录好之后才开始下一段，不会有overwriting
    sd.wait()

    # 将录音保存下来建立音频文件
    wv.write(f"./recordings/{filename}.wav", recording, freq, sampwidth=2)
    # 将最近生成的文件开始读
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
        
        # 这是一个用于分辨是否为静音或杂音的判断
        if result.no_speech_prob < 0.5:
            print(result.text)
            # 小文字入txt文件
            with open(config.TRANSCRIPT_FILE, 'a') as f:
                f.write(result.text)

            # 过滤掉之前已经读过的文件
            transcribed.append(latest_recording)
        
        if '幫我總結' in result.text: # 进行一个触发，这个‘帮我总结’可以替换成其他你想要的。需要注意的是，whisper默认识别中文后出来的是繁体字，所以这个触发语句如果是中文的话也要是繁体字哦
            print("--------Deploying Jarvis--------")
            transcript = open('./transcriptions/transcript.txt', 'r').read()
            print(GPT(transcript))


