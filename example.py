from pytorch_model import SoundNet8_pytorch
import torch, torchaudio
torchaudio.set_audio_backend("soundfile")

audio_path = "./SI1657.WAV"

if __name__=="__main__":
    model = SoundNet8_pytorch()
    model.load_state_dict(torch.load("./sound8.pth"))
    wav, sr = torchaudio.load(audio_path)
    print(wav.shape)

    wav = wav.unsqueeze(1).unsqueeze(-1).repeat(1,1,8,1) # errors occur when the wav is too short
    feats = model.extract_feat(wav)
    # features for layer1 to layer8
    for idx, f in enumerate(feats):
        print(f"feature shape for layer {idx}: {f.shape}")