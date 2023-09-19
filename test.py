url = 'https://github.com/hhoppe/data/raw/main/video.mp4'
import mediapy as media
video = media.read_video(url)
print(video.shape, video.dtype)  # It is a numpy array.
print(video.metadata.fps)  # The 'metadata' attribute includes framerate.
media.show_video(video)  # Play the video using the retrieved framerate.

media.show_images(video, height=80, columns=4)  # Show frames side-by-side.

video = media.moving_circle((128, 128), num_images=10)
media.show_video(video, fps=10)

media.write_video('./video.mp4', video, fps=60)

# Darken a video frame-by-frame:
filename_in = '/tmp/video.mp4'
filename_out = '/tmp/out.mp4'
with media.VideoReader(filename_in) as r:
    print(f'shape={r.shape} fps={r.fps} bps={r.bps}')
    darken_image = lambda image: media.to_float01(image) * 0.5
    with media.VideoWriter(
            filename_out, shape=r.shape, fps=r.fps, bps=r.bps) as w:
        for image in r:
            w.add_image(darken_image(image))
media.show_video(media.read_video(filename_out), fps=60)