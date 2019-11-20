# coding: utf-8
from __future__ import division, print_function
import os,subprocess
import speech_recognition as sr
from os import path
from pydub import AudioSegment


import models
import data

import theano
import sys
from io import open

import theano.tensor as T
import numpy as np
import sys
from io import open
from data import EOS_TOKENS, PUNCTUATION_VOCABULARY

def convert_video_to_audio(input_file_path):
	file_name = os.path.basename(input_file_path)
	if 'mouthcropped' not in file_name:
		raw_file_name = os.path.basename(file_name).split('.')[0]
		file_dir = os.path.dirname(input_file_path)
		file_path_output = file_dir + '/' + raw_file_name + '.wav'
		print('processing file: %s' % input_file_path)
		#subprocess.call(['ffmpeg', '-i', input_file_path, '-codec:a', 'pcm_s16le', '-ac', '1', file_path_output])
		subprocess.call(['ffmpeg', '-i', input_file_path, '-f', 'mp3', '-ab', '192000', file_path_output])
		print('file %s saved' % file_path_output)
		print("MP4 converted to MP3.......")
		return file_path_output


def convert_file_to_wav(input_file_path):
	# convert mp3 file to wav    
	print(input_file_path)                                                   
	sound = AudioSegment.from_mp3(input_file_path)
	filename = os.path.basename(input_file_path).split('.')[0]
	output_file_path = os.path.dirname(input_file_path) + "/" +filename + ".wav"
	sound.export(output_file_path, format ="wav")
	print(" FILE CONVERTED TO .wav")
	return output_file_path

def trim_audio(input_file, output_dir, time = 59):
    if output_dir[-1] != '/':
        output_dir += '/'
    print("TRIMMING AUDIO.....")
    suffix = input_file.split('.')[-1]
    suffix ='wav'
    filename = input_file.split('/')[-1].split('.')[0]
    os.makedirs(output_dir+'/'+filename,exist_ok=True)
    input_file = input_file.strip().replace(" ", "\\ ") 
    _status = subprocess.call(["ffmpeg -n -i  {0} -f  segment -segment_time {1} {2}/{3}/aud_%03d.wav".format(input_file, time, output_dir,filename.replace(" ","\\ "))], shell=True)
    output_file_path = os.path.dirname(input_file) + '/' + filename
    print("Audio Folder Location: ", output_file_path)
    return output_file_path

def convert_audio_to_text (audio_directory):
	print("Converting Audio to Text...........")
	file = []
	for f in os.listdir(audio_directory):
		file.append(f)
		files = sorted(file)
	text_file_location = os.path.dirname(audio_directory)
	text_file_name = os.path.basename(audio_directory)
	output_file = audio_directory + "/" + text_file_name + "_text.txt"
	text_file = open(output_file,"a")
	r = sr.Recognizer()
	for file in files:
		data = sr.AudioFile( audio_directory + '/' + file )
		with data as source:
			r.adjust_for_ambient_noise(source, duration = 0.5)
			audio = r.record(source)
		text_file.write(r.recognize_google(audio))
		text_file.write(" ")
		#print(r.recognize_google(audio))
		print(" . ")
		print("  ")
	print("Text file saved at ", output_file)
	text_file.close()
	return output_file
MAX_SUBSEQUENCE_LEN = 200

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore_with_pauses(output_file, text, pauses, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]
            subsequence_pauses = pauses[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict_function(to_array(converted_subsequence), to_array(subsequence_pauses, dtype=theano.config.floatX))

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

def restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict_function(to_array(converted_subsequence))

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

def extract_text(file):
	#convert audio to wav
	if (os.path.basename(file).split('.')[1]) == "mp4":
		file = convert_video_to_audio(file)

	# convert mp3 to wav
	if (os.path.basename(file).split('.')[1]) == "mp3":
		file = convert_file_to_wav(file)
	
	# trim audio to 1min length and store them in a separate folder
	audio_directory = trim_audio(file,os.path.dirname(file))
	
	
	# connvert audios to text
	text_file = convert_audio_to_text(audio_directory)

	#punctuate text

	model_file = "/home/stuti/Documents/PBL/INTERSPEECH-T-BRNN.pcl"

	output_file = os.path.dirname(text_file)+"/output"+os.path.basename(text_file)

	use_pauses = len(sys.argv) > 3 and bool(int(sys.argv[3]))

	x = T.imatrix('x')

	if use_pauses:

		p = T.matrix('p')
		print("Loading model parameters...")
		net, _ = models.load(model_file, 1, x, p)

		print("Building model...")

		predict = theano.function(
			inputs=[x, p],
			outputs=net.y
			)
	else:
		print("Loading model parameters...")
		net, _ = models.load(model_file, 1, x)

		print("Building model...")
		predict = theano.function(
			inputs=[x],
			outputs=net.y
			)
	word_vocabulary = net.x_vocabulary
	punctuation_vocabulary = net.y_vocabulary

	reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
	reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

	input_text = open(text_file).read()

	text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
	pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

	if not use_pauses:
		restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, predict)
	else:
		if not pauses:
			pauses = [0.0 for _ in range(len(text)-1)]
		restore_with_pauses(output_file, text, pauses, word_vocabulary, reverse_punctuation_vocabulary, predict)

	final_output_file = os.path.dirname(text_file)+'/final_output'+os.path.basename(text_file)

	with open(output_file, 'r', encoding='utf-8') as in_f, open(final_output_file, 'w', encoding='utf-8') as out_f:
		last_was_eos = True
		first = True
		for token in in_f.read().split():
			if token in PUNCTUATION_VOCABULARY:
				out_f.write(token[:1])
			else:
				out_f.write(('' if first else ' ') + (token.title() if last_was_eos else token))

			last_was_eos = token in EOS_TOKENS
			if last_was_eos:
				out_f.write('\n')
				first = True
			else:
				first = False

	return final_output_file

if __name__ == '__main__':

	file = "/home/stuti/Documents/PBL/voice2_insights/files_uploaded"

	text_file = extract_text(file)


		