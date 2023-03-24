'''
这个代码利用了openai的whisper模型和gpt3.5模型
帮助用户将MP3文件转成文字，并利用gpt模型的强大语言功能进行对文字的总结
'''

import openai
import tiktoken
import numpy as np

openai.api_key = open("C:\\Users\\tom96\\PycharmProjects\\pythonProject3\\key.txt", "r").read()

token_breaker = 3000 #建议设置成3000或以下, gpt3.5模型目前只支持4096个token数量。
encoding_break_status = False #检测我们的文字是否需要切分成小段，如果需要，则此状态打开

'''
以下函数用于处理长度大于token_breaker的文字。
将不同的片段分离成一个numpy矩阵，以备后用。

还有，不要忘记还没有添加到矩阵中的rest_content，因为大小不同所以放不进去。
'''
def encoding_break(transcript):
    global encoding_break_status
    num = len(transcript) // token_breaker
    array = np.zeros((num,token_breaker), dtype=np.int64)
    remainder = len(transcript) % token_breaker
    if remainder > 0:
        rest_content = np.zeros((1, remainder), dtype=np.int64)
        rest_content[0] = transcript[num*token_breaker:]
    for i in range(num):
        a = i*token_breaker
        c = (i+1)*token_breaker
        array[i:]=transcript[a:c]
    encoding_break_status = True
    return array, rest_content


'''
以下代码用到whisper模型将MP3文件转换成文字
'''
audio_file= open("YOUR MP3 FILE LOCATION", "rb")
print('----------Starting your conversion from MP3 to transcript----------')
transcript = openai.Audio.transcribe("whisper-1", audio_file)
transcript = transcript.text

'''
OpenAI使用tiktoken包将您的脚本转换为token。
我们需要用这个来计算和分离我们的文字，以传递给gpt-3.5模型，
因为该模型最多只允许4096个token！
'''
print('----------Transcript received from OpenAi Whisper Model----------')
encode = tiktoken.encoding_for_model('gpt-3.5-turbo')
encode_list = encode.encode(transcript)
print(f"The original token numbers for the transcript is {len(encode_list)} tokens")

'''
检查我们的总文本是否超过token的限制，如果是的话我们就把它切开放在一个matrix里面
如果没达到上限，我们就直接喂给模型。
需要注意的是，模型回复给你的也是计算token的，所以不要把token_breaker这个值设置太大
'''
if len(encode_list) > token_breaker:
    final_list = encoding_break(encode_list)[0]
    remain_content = encoding_break(encode_list)[1]
    print("----------Separation process initiated----------")
else:
    final_list = np.zeros((1,len(encode_list)), dtype=np.int64)
    final_list[0] = encode_list
    print("----------No separation needed----------")


message_history = [{"role": "assistant", "content": f"OK"}]

'''
下面利用到gpt模型，我们将一个个片段分开喂给gpt模型
'''
def GPT(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"Summarize in 300 words: {input}"}) # 这完全取决于你想让gpt怎么回复，如果你想让他给你重要的点一点点写下来bullet points也可以，你让他直接给你翻译成阿拉伯文都可以。
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # 我觉得还是把之前gpt的回复放进来，他能更好理解上下文
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

'''
最后我们把一个个总结的小片段拼凑成一个大片段
'''
final_sum = 'Hi here is your final summarization\n'
if encoding_break_status is True: #Check to see if we used the encoding_break function, if true, we process transcripts one by one. Otherwise, we just feed the original transcript.
    for i in range(len(final_list)):
        print(f'--------Processing the : {i + 1} paragraph--------')
        response = (GPT(encode.decode(final_list[i])))
        final_sum += response
    print(f'--------Processing the last paragraph--------')
    final_sum += GPT(encode.decode(remain_content[0]))
else:
    print(f'--------Processing the transcript--------')
    final_sum +=GPT(encode.decode(final_list[0]))

print(final_sum)
