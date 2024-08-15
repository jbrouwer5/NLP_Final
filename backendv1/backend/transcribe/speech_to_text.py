def transcribe(pipe, audio_path:str) -> str:
    

    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]

    # - result = pipe(sample)
    # + result = pipe("audio.mp3")

    # result = pipe(sample, generate_kwargs={"language": "english"})
    # result = pipe(sample, generate_kwargs={"task": "translate"})
    print("\n\nDOING PIPE\n\n\n")
    result = pipe(audio_path, generate_kwargs={"language": "english"})
    print("\n\nFINISHED PIPE\n\n\n")
    return result["text"]