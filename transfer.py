from model import get_input_optimizer, get_style_model_and_losses


def run_style_transfer(cnn, content_img, style_img1, style_img2, input_img, num_steps=300,
                       style_weight=10000, content_weight=1, tv_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses, tv_losses = get_style_model_and_losses(cnn, style_img1, style_img2,
                                                                                content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values
            # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0.05, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0
            tv_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for tvl in tv_losses:
                tv_score += tvl.loss

            # взвешивание ощибки
            style_score *= style_weight
            content_score *= content_weight
            tv_score *= tv_weight

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} TotalVariation Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), tv_score.item()))
                print()

                input_img.data.clamp_(0, 1)

            return style_score + content_score + tv_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img
