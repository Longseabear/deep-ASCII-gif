from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import numpy as np
import skimage.color as color
import imageio
from skimage.transform import resize
from scipy import signal
import tqdm
from numpy.lib.stride_tricks import as_strided


def min_max_normal(x):
    return (x-np.min(x))/(np.max(x) - np.min(x))

import argparse
parser = argparse.ArgumentParser(description='amazing-ascii gif to ascii.gif')
parser.add_argument('--src', default='gif/input.gif', help='only gif image')
parser.add_argument('--patch_size', default=(3,3), help='image patch size')
parser.add_argument('--output_word', default=(10,10), help='one word size')
parser.add_argument('--output_img_size', default=(240,320), help='output size(resize)')
parser.add_argument('--input_str', default="0123456789 ", help='component string')
parser.add_argument('--font_path', default="./font/Consolas.ttf", help='font!')
parser.add_argument('--output_path', default="./output/out.gif", help='output path')

args = parser.parse_args()
input_str = args.input_str
patch_size = args.patch_size
output_word = args.output_word
output_img_size = args.output_img_size

''':
ref: https://gist.github.com/arifqodari/dfd734cf61b0a4d01fde48f0024d1dc9
'''
def strided_convolution(image, weight, stride):
    im_h, im_w = image.shape
    f_h, f_w = weight.shape

    out_shape = (1 + (im_h - f_h) // stride[0], 1 + (im_w - f_w) // stride[1], f_h, f_w)
    out_strides = (image.strides[0] * stride[0], image.strides[1] * stride[1], image.strides[0], image.strides[1])
    windows = as_strided(image, shape=out_shape, strides=out_strides)

    return np.tensordot(windows, weight, axes=((2, 3), (0, 1)))

def font2img(ch):
    unicode_text = u"{}".format(ch)
    font = ImageFont.truetype(args.font_path, 28, encoding="unic")
    text_width, text_height = font.getsize(unicode_text)
    canvas = Image.new('RGB', (text_width, text_height), "white")

    draw = ImageDraw.Draw(canvas)
    draw.text((0,0), u'{}'.format(ch), 'black', font)
    return np.array(canvas)/255.

def main():
    chars = set([i for i in input_str])
    out = []
    mapping_img = {}
    for i in chars:
        mapping_img[i] = color.rgb2gray(resize(font2img(i), output_word))
        b = np.mean(mapping_img[i])
        out.append((b, i))

    out = sorted(out)
    total_ascii_num = len(out)
    mapping = {}
    for i in range(len(out)):
        mapping[i] = out[i][1]

    import imageio
    gif = imageio.get_reader(open("./gif/input.gif", 'rb').read(), '.gif')

    # Here's the number you're looking for
    number_of_frames = len(gif)

    weight = np.ones(patch_size)

    images = []
    for frame in tqdm.tqdm(gif):
        f = color.rgb2gray(frame[:, :, :3])
        f = min_max_normal(f)

        means = strided_convolution(f, weight, patch_size) / (patch_size[0] * patch_size[1])
        h, w = means.shape
        out = np.zeros((h * output_word[0], w * output_word[1]))
        for i in range(h):
            for j in range(w):
                b = int(means[i, j] * total_ascii_num - 0.000001)
                out[i * output_word[0]:i * output_word[0] + output_word[0],
                j * output_word[1]:j * output_word[1] + output_word[1]] = mapping_img[mapping[b]] * 255.

        #    out[y:y+patch_size[0], x:x+patch_size[1]] = color.rgb2gray(resize(font2img(mapping[b]), patch_size))
        out = resize(out, output_img_size)
        images.append(out.astype(np.uint8))
    imageio.mimsave(args.output_path, images, fps=55)

if __name__ == '__main__':
    main()