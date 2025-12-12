from torchcodec.decoders import VideoDecoder

device = "cpu"  # or e.g. "cuda" !
decoder = VideoDecoder("path/to/video.mp4", device=device)

decoder.metadata
# VideoStreamMetadata:
#   num_frames: 250
#   duration_seconds: 10.0
#   bit_rate: 31315.0
#   codec: h264
#   average_fps: 25.0
#   ... (truncated output)

# Simple Indexing API
decoder[0]  # uint8 tensor of shape [C, H, W]
decoder[0 : -1 : 20]  # uint8 stacked tensor of shape [N, C, H, W]

# Indexing, with PTS and duration info:
decoder.get_frames_at(indices=[2, 100])
# FrameBatch:
#   data (shape): torch.Size([2, 3, 270, 480])
#   pts_seconds: tensor([0.0667, 3.3367], dtype=torch.float64)
#   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)

# Time-based indexing with PTS and duration info
decoder.get_frames_played_at(seconds=[0.5, 10.4])
# FrameBatch:
#   data (shape): torch.Size([2, 3, 270, 480])
#   pts_seconds: tensor([ 0.4671, 10.3770], dtype=torch.float64)
#   duration_seconds: tensor([0.0334, 0.0334], dtype=torch.float64)