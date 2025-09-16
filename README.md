## Project goal:
### Turning (recorded) audio to keystrokes
```
A proof of concept on how your mic and screensharing can be used against you
```

#### What I hope to get more familiar with and learn:
- modern ML training 
    - CNNs, transfer learning, embeddings, cluster, ???
    - my experience in ML training is close to a decade old at this point
- audio I/O
- audio feature extraction
- digital signal processing (DSP) 
    - mainly envelope and spectral flux (maybe others? maybe not these?)
- a reminder to step away from ai tools doing any work because this is new and interesting to learn
- some of that math I have a minor in that I probably forget

#### Loose roadmap:
1. Take an audio file of someone typing and (only) detect and segment individual keystroke events
2. then cluster/classify those events by acoustic class (e.g., hard click vs soft click, different keyboard types, or “key vs non-key”)
3. produce visualizations of the audio data 
4. build a small demo UI that shows event timestamps and spectrogram snippets. 
5. then do the ML on that data since we know it's good
6. with that trained model, feed it an audio file and see what it outputs vs what the expected output is
7. ????
8. profit


#### Ethics part:
```
Yes this could be used for all kinds of malicious purposes, but it's a fun project so I'm not gonna worry about it for now.

this is just for experimental purposes using my own typing data and recordings. 

If you take anything from this repo, you bear the consequences of your actions.
``` 


#### Notes:
- **Overfitting to room/mic:** models might learn microphone or room signature rather than keystroke quality. I don't really care too much since this is a PoC. To mitigate that, I'd have a different approach with augmentation, different microphones, keyboards, and cross-person splits.
- **golden path/futuristic vision:** I think if vibe-coding really worked this thing would have a web interface to segment audio thumbnails and assign characters to them in real time. That way labeling could happen in real time and you could maybe decode live audio.
- **amplitude envelope**: the amplitude envelope is basically the changes in the loudness or intensity of a sound over time. detecting these can help find the keystroke events



### Setup notes for future me
- I'm on an M1 mac
- I did some light research and it seems like `miniforge`/`mamba` is smoother with `PyTorch`/`TensorFlow` and audio libraries compared to `venv`
- ```
  # download & install Miniforge (Apple Silicon build)
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  bash Miniforge3-MacOSX-arm64.sh
  
  # restart shell or run:
  source ~/miniforge3/bin/activate
  # now you should see a ("base" prefix)

  # creating a new environment for audio work
  mamba create -n audio python=3.11
  mamba activate audio
  
  #install some utilities
  mamba install jupyterlab matplotlib numpy scipy
  ```
- run `python3 test-seupt.py` to see if things are good to go
- switch to jupyter with `jupyter lab`
- things work so `mamba env export > environment.yml`