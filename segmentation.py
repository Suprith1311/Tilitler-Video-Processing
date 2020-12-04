import cv2
from enum import Enum
import numpy as np
import argparse
import sys
import os
import ntpath


class ProcessingMethod(Enum):

    SIMPLE = 0

    @staticmethod
    def EnumFromString(method):

        if method in ('simple', 'SIMPLE'):
            return ProcessingMethod.SIMPLE
        else:
            raise NotImplementedError


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--video_file_path', type=str, help='the video path that will be played')
    parser.add_argument('-m', '--method', type=str, help='the processing method that will be employed')
    parser.add_argument('-t', '--video_target_path', type=str, help='the video path that will be played')

    return parser.parse_args(argv)


def writeFramesToFile(frames, filename):

    try:
        height, width, h = np.shape(frames[0])
        vid = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MP4V'), 15, (width, height))

        print('Writing frames ...')
        for i, frame in enumerate(frames):
            vid.write(frame)
        print('Done writing frames')

        cv2.destroyAllWindows()
        vid.release()
    except Exception as e:
        print('An error occured while saving the semented frames', e)


def simpleProcessing(frames):

    # gray frames all 3 channels
    g_frames = [np.dstack(
        (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
                for f in frames]

    # frame reference is the 10-th frame
    # compute the mean of the first frame to have a better estimate of the background
    ref_frame = np.mean(g_frames[:30], axis=0)

    # generate the masks
    masks = np.abs(np.subtract(ref_frame, g_frames))

    thresholded_masks = []

    # remove pixel fluctuation delta = 20 pixels
    thresholded_masks = np.ma.getmask(np.ma.masked_greater(masks, 20))
    thresholded_masks = thresholded_masks.astype(int)

    frames = np.array(frames).astype(float)

    # then multiply the original images with the masks to get the segmentation
    result = np.multiply(thresholded_masks, frames)
    return np.uint8(result)


def ProcessVideo(filename, destFile, method):
    '''Carries out the segmentation of a video file according to the selected processing method.'''
    try:
        # load de the file
        video = cv2.VideoCapture(filename)

        if video.isOpened():
            # number total of frames
            n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            print('Reading the frames ...')
            frames = []
            for n in range(n_frames):
                ret, frame = video.read()
                frames.append(frame)
            print('Done reading the frames')

        print('Applying the processing method:', str(method), str('...'))
        new_frames = []
        if method == ProcessingMethod.SIMPLE:
            new_frames = simpleProcessing(frames)



        print('Done applying the processing method:', str(method))
        # save the segmented result to a file
        if len(new_frames) != 0:
            writeFramesToFile(new_frames, destFile)

        # release resources
        video.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print('An error occurred while reading the video file:', e)


def main(args):



    if args.video_file_path is None:
        print('Please provide a video path !')
        return

    # compose the target dir
    dir_name = ntpath.dirname(args.video_file_path)
    only_fname = ntpath.basename(args.video_file_path)

    target_p = dir_name + str('/') + only_fname.split('.')[0] + str(
        '_processed.mp4') if args.video_target_path is None else args.video_target_path
    method = ProcessingMethod.SIMPLE if args.method is None else ProcessingMethod.EnumFromString(args.method)

    # process the video
    ProcessVideo(args.video_file_path, target_p, method)
    # playVideo(args.video_file_path, res, fps, mono)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
