def check_audio_format(audio) -> str:
    if isinstance(audio, bytes):
        audio_bytes = audio
    elif isinstance(audio, str):
        with open(audio, "rb") as f:
            audio_bytes = f.read()
    if audio_bytes.startswith(b'RIFF') and audio_bytes.find(b'WAVE') != -1:
        return 'wav'
    elif audio_bytes.startswith(b'OggS'):
        return 'ogg'
    else:
        raise NotImplementedError()
