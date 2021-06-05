"""
Utility functions for landmark-based human action recognition using path signatures
"""

import requests

import cv2
from moviepy.editor import VideoFileClip, clips_array
import numpy as np
from tqdm.notebook import tqdm

def download(source_url, target_filename, chunk_size=1024):
    """
    Download a file via HTTP.

    Parameters
    ----------
    source_url: str
        URL of the file to download
    target_filename: str
        Target filename
    chunk_size: int
        Chunk size
    """
    response = requests.get(source_url, stream=True)
    file_size = int(response.headers['Content-Length'])

    with open(target_filename, 'wb') as handle:
        for data in tqdm(response.iter_content(chunk_size=chunk_size),
                         total=int(file_size / chunk_size), unit='KB',
                         desc='Downloading dataset:'):
            handle.write(data)

class SkeletonFace():
    """
    Skeleton representation for visualisation and animation consisting of dots
    representing landmarks and lines representing connections.
    """

    # When transforming key points, use a fraction of the image size as a minimum of empty,
    # black region around the skeleton
    IMG_BORDER_FRAC = 0.1

    def __init__(self,
                 target_width,
                 target_height,
                 radius=10,
                 confidence_coord=None,
                 transform_keypoints=False,
                 n_points_per_image = 68):
        """
        Parameters
        ----------
        target_width: int
            Image width for visualisation
        target_height: int
            Image height for visualisation
        radius: int
            Landmark radius
        confidence_coord: int
            Index of confidence estimates
        draw_connections: bool
            Whether to draw connections between the key points
         transform_keypoints: bool
            Whether to transform key points
        """
        self.n_points_per_image = n_points_per_image
        self.landmarks = [(226, 0, 255)] * 17 + [(155, 0, 255)] * 5 + [(198, 0, 255)] * 5 + [(70, 0, 255)] * 4 + \
                         [(141, 255, 0)] * 5 + [(99, 255, 0)] * 6 + [(127, 0, 255)] * 6 + [(255, 42, 0)] * 12 + \
                         [(255, 113, 0)] * 8
        
        self.radius = radius
        self.confidence_coord = confidence_coord
        self.target_width = target_width
        self.target_height = target_height
        self.transform_keypoints = transform_keypoints

    def draw(self, keypoints):

        """
        Plot a static keypoint image.

        Parameters
        ----------
        keypoints: numpy array
            Keypoints to be drawn
        """
        img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        keypoints = keypoints[:, 0:2].copy()
        if self.transform_keypoints:
            keypoints = self._transform_keypoints(keypoints)

        keypoints = np.around(keypoints).astype(np.int32)

        self._draw_landmarks(keypoints, img)

        return img

    def _draw_landmarks(self, keypoints, img):
        for i, colour in enumerate(self.landmarks[0:self.n_points_per_image]):
            # Skip points with confidence 0 (usually means not detected)
            if self.confidence_coord is not None and keypoints[i, self.confidence_coord] == 0:
                continue
            point = tuple(keypoints[i])
            if point[0] != 0 or point[1] != 0:
                cv2.circle(img, point[0:2], self.radius, colour, -1)

    def _transform_keypoints(self, keypoints):
        keypoints -= np.amin(keypoints, axis=0)
        keypoint_scale = np.amax(keypoints, axis=0)
        keypoints *= np.array((self.target_width, self.target_height)) * (
            1 - 2 * SkeletonFace.IMG_BORDER_FRAC) / keypoint_scale
        keypoints += np.array((self.target_width, self.target_height)) * SkeletonFace.IMG_BORDER_FRAC

        return keypoints

    def animate(self, keypoints, filename=None, fps=25, codec='XVID'):
        """
        Convert key points to a animation and output to a video file.

        Parameters
        ----------
        keypoints: numpy array
            Array of keypoints in the form [frame,landmark,coords]
        filename: string, optional (default is None)
            If given the video is saved to the specified file
        fps: float
            Number of frames per second
        codec: str
            Video codec represented in fourcc format
        """
        if filename is not None:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vid_file = cv2.VideoWriter(filename, fourcc, fps,
                                       (self.target_width, self.target_height))

        vid = np.zeros((keypoints.shape[0], self.target_height, self.target_width, 3),
                       dtype=np.uint8)

        for i in range(vid.shape[0]):
            vid[i] = self.draw(keypoints[i])
            if filename is not None:
                frame = cv2.cvtColor(vid[i], cv2.COLOR_RGB2BGR)
                vid_file.write(frame)
        if filename is not None:
            vid_file.release()

        return vid

def generate_one_clip(keypoints, clip_height=768, display_height=240,
                  temp_file='__temp__.avi', word = '', transform_keypoints = False):
    """
    Display a side-by-side animation comprising the skeleton and (optionally) the source
    video.

    Parameters
    ----------
    keypoints : numpy array
        Array of keypoints in the form [frame,landmark,coords]
    clip_height : int
        Desired clip height (applies to both source video and skeleton, the former is upscaled)
    display_height : int
        Desired display height
    temp_file : int
        Temporary file for transcoding
    include_source_video: bool
        Whether to include the source video
    word: str,
        The word that the person says
    """
    keypoints = np.copy(keypoints) 
    rescaling_factor = clip_height/display_height
    keypoints = keypoints * rescaling_factor
    n_points_per_image = len(keypoints[0])
    
    #duration = 1
    skeleton = SkeletonFace(target_width=clip_height, target_height=clip_height, transform_keypoints = transform_keypoints,
                            n_points_per_image = n_points_per_image)
    skeleton.animate(keypoints, temp_file, )#fps=len(keypoints)/duration)
    
    clip_skeleton = VideoFileClip(temp_file)
    
    return clip_skeleton

def display_animation_multiple_faces(keypoints_array, clip_height=768, display_height=240,
                      temp_file='__temp__.avi', word = '', transform_keypoints = False):
    clip_list = []
    for i, keypoints in enumerate(keypoints_array):
        clip = generate_one_clip(keypoints, clip_height = clip_height, display_height = display_height,
                                 temp_file = temp_file.replace('__.', '__{}.'.format(i)), word = word,
                                 transform_keypoints = transform_keypoints)
        clip_list.append(clip)
    clip_final = clips_array([clip_list])
    return clip_final.ipython_display(height=display_height, rd_kwargs=dict(logger=None))   

def display_animation_face(keypoints, clip_height=768, display_height=240,
                      temp_file='__temp__.avi', word = '', transform_keypoints = False):
    """
    Display a side-by-side animation comprising the skeleton and (optionally) the source
    video.

    Parameters
    ----------
    keypoints : numpy array
        Array of keypoints in the form [frame,landmark,coords]
    clip_height : int
        Desired clip height (applies to both source video and skeleton, the former is upscaled)
    display_height : int
        Desired display height
    temp_file : int
        Temporary file for transcoding
    include_source_video: bool
        Whether to include the source video
    word: str,
        The word that the person says
    """

    if len(word) > 0:
        print(word)
    clip = generate_one_clip(keypoints, clip_height = clip_height, display_height = display_height,
                             temp_file = temp_file, word = word, 
                             transform_keypoints = transform_keypoints)

    return clip.ipython_display(height=display_height, rd_kwargs=dict(logger=None))
