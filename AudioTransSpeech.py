'''
We can use the following code to transfer an AUDIO file in to transcript,
then ask the gpt language model to help us summarize the transcript.
If you are at a boring meeting, record it and dump it in here to form bullet points.
If you are trying to learn stuff from YouTube and don't have the time to watch the whole video,
dump it in here to generate summary.
'''

import openai
import tiktoken
import numpy as np

openai.api_key = open("C:\\Users\\tom96\\PycharmProjects\\pythonProject3\\key.txt", "r").read()

token_breaker = 3000 #Recommended to set below 3000, Openai gpt-3.5 model only accepts 4096 tokens, so we are going to break the text at a given token limit
encoding_break_status = False #Determine whether the script is long enough to using the encoding_break function

'''
The following function is used to process transcript that is larger than the token_breaker.
Separate different pieces into a numpy matrix for later use

And, don't forget the rest_content that hasn't been added to the matrix since they are not the same size.
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
Following code is used to convert your mp3 files into transcript.
'''
audio_file= open("YOUR MP3 FILE LOCATION", "rb")
print('----------Starting your conversion from MP3 to transcript----------')
transcript = openai.Audio.transcribe("whisper-1", audio_file)
transcript = transcript.text

'''
OpenAI use tiktoken package to convert your script into tokens
We need this to calculate and separate the transcript to pass it down to gpt-3.5 model
which only allowed for 4096 tokens MAX!
'''
print('----------Transcript received from OpenAi Whisper Model----------')
encode = tiktoken.encoding_for_model('gpt-3.5-turbo')
encode_list = encode.encode(transcript)
print(f"The original token numbers for the transcript is {len(encode_list)} tokens")

'''
Check to see if the transcript is over our token limit, if it is, slice it up to form a new numpy matrix
if it is within the token limit, put it as it is in a matrix.
Something to notice here, is that we set the token limit, not the original 4096 token limit,
because to acquire responses consumes token as well, thus do not set the token_breaker larger than 3000
'''
if len(encode_list) > token_breaker:
    final_list = encoding_break(encode_list)[0]
    remain_content = encoding_break(encode_list)[1]
    print("----------Separation process initiated----------")
else:
    final_list = np.zeros((1,len(encode_list)), dtype=np.int64)
    final_list[0] = encode_list
    print("----------No separation needed----------")

'''
Starting to pass the arguments into the GPT model
What we trying to do here is to pass it one time, summarize what we have.
Use the summarization and the next paragraph together and to put it into summarization again.

Since a matrix can only fit in a fixed number of columns, so
Do not forget we still have the remaining content.
'''

message_history = [{"role": "assistant", "content": f"OK"}]

'''
For the following function, we pass our transcripts(sliced up) in the matrix into the gpt-3.5 models one by one
'''
def GPT(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"Summarize in 300 words: {input}"}) # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=prompt_history
    )
    print(f"{completion.usage.total_tokens} tokens consumed.")
    reply_content = completion.choices[0].message.content
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

'''
We can then sum up all the processed summarization to a str(final_sum)
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

