
import argparse

from loader import Images, unloader
from model import model
from transfer import run_style_transfer


def run(args):
    images = Images(args.content, args.style1, args.style2)
    input_img = images.content.clone()
    result = run_style_transfer(model, images.content, images.style1, images.style2, input_img, num_steps=args.iteration)
    result = unloader(result[0])
    result.save(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural style transfer.")
    parser.add_argument("--content", type=str,
                        help="Path to content image")
    parser.add_argument("--style1", type=str,
                        help="Path to first style image")
    parser.add_argument("--style2", type=str,
                        help="Path to second parse image")
    parser.add_argument("--output", default='out.jpg',
                        help="Path to second parse image")
    parser.add_argument("-i", "--iteration", type=int, default=300,
                        help="Number of iterations")

    args = parser.parse_args()

    run(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
