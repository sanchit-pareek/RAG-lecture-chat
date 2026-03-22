import os
import subprocess
files = os.listdir('videos')
for file in files:
    if file.endswith('.mp4'):
        print(file)
        lecture_no = file.split('c')[1].split('_')[0]
        print(lecture_no)
        subprocess.run(["ffmpeg", '-i', f'videos/{file}', f'audios/lec{lecture_no}.mp3'])